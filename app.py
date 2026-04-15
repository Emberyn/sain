import os
import time
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import io
import zipfile
import json

from src.config import Config
from src.models import EdgeModel, InpaintingModel
from edge_make.edge_extraction_2 import SobelConv

app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

app.config['PERMANENT_SESSION_LIFETIME'] = 300

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint', 'config.yml')
config = Config(config_path)
config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading models to {config.DEVICE}...")

# 初始化并加载模型
edge_model = EdgeModel(config).to(config.DEVICE)
inpaint_model = InpaintingModel(config).to(config.DEVICE)

edge_model.load()
inpaint_model.load()

edge_model.eval()
inpaint_model.eval()

# 初始化边缘检测算子
sobel_conv = SobelConv(device=config.DEVICE, edge_to_binary=False)
for param in sobel_conv.parameters():
    param.requires_grad = False

print("Models loaded successfully and ready for inference!")

def preprocess_image(image_bytes, size=256):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((size, size))
    img_np = np.array(img)
    return img_np

def preprocess_mask(mask_bytes, size=256):
    mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
    mask = mask.resize((size, size))
    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.float32) # 1 for holes
    return mask_np

def to_tensor(img_np):
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=2)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
    if img_t.max() > 1.0:
        img_t /= 255.0
    return img_t

def tensor_to_base64(tensor):
    img_np = tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img = Image.fromarray(img_np, mode='L')
    else:
        img = Image.fromarray(img_np, mode='RGB')
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_single_image(image_bytes, mask_bytes, config, edge_model, inpaint_model, sobel_conv):
    """
    处理单张图片的修复逻辑（从原/api/repair提取）
    """
    img_np = preprocess_image(image_bytes, size=config.INPUT_SIZE)
    mask_np = preprocess_mask(mask_bytes, size=config.INPUT_SIZE)

    gray_np = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    images = to_tensor(img_np).unsqueeze(0).to(config.DEVICE)
    masks = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
    gray_img = torch.from_numpy(gray_np).unsqueeze(0).unsqueeze(0).to(config.DEVICE) / 255.0

    masks_3c = torch.cat([masks, masks, masks], dim=1)

    start_time = time.time()

    with torch.no_grad():
        img_for_sobel = images.clone()
        edge_prob = sobel_conv(img_for_sobel)
        edge = torch.cat([edge_prob, edge_prob, edge_prob], dim=1)

        outputs_edge = edge_model(images, masks_3c, edge, gray_img)
        outputs_edge = (outputs_edge * masks_3c) + (edge * (1 - masks_3c))

        outputs_img = inpaint_model(images, masks_3c, outputs_edge, gray_img)
        outputs_merged = (outputs_img * masks_3c) + (images * (1 - masks_3c))

    end_time = time.time()
    inference_time_ms = int((end_time - start_time) * 1000)

    repaired_b64 = tensor_to_base64(outputs_merged)
    edge_b64 = tensor_to_base64(outputs_edge)
    original_edge_b64 = tensor_to_base64(edge)

    return {
        'repaired_image': f"data:image/png;base64,{repaired_b64}",
        'edge_image': f"data:image/png;base64,{edge_b64}",
        'original_edge': f"data:image/png;base64,{original_edge_b64}",
        'inference_time_ms': inference_time_ms
    }

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/repair', methods=['POST'])
def repair():
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Missing image or mask'}), 400

        image_file = request.files['image']
        mask_file = request.files['mask']

        result = process_single_image(
            image_file.read(),
            mask_file.read(),
            config,
            edge_model,
            inpaint_model,
            sobel_conv
        )

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/repair_batch', methods=['POST'])
def repair_batch():
    """
    批量处理多张图片（最多20张）
    """
    try:
        if 'images' not in request.files or 'masks' not in request.files:
            return jsonify({'error': '缺少图片或遮罩文件'}), 400

        images_files = request.files.getlist('images')
        masks_files = request.files.getlist('masks')

        if len(images_files) != len(masks_files):
            return jsonify({'error': '图片和遮罩数量不匹配'}), 400

        if len(images_files) > 20:
            return jsonify({'error': '最多只能上传20张图片'}), 400

        if len(images_files) == 0:
            return jsonify({'error': '未接收到任何图片'}), 400

        results = []
        total_start_time = time.time()

        for i, (image_file, mask_file) in enumerate(zip(images_files, masks_files)):
            try:
                result = process_single_image(
                    image_file.read(),
                    mask_file.read(),
                    config,
                    edge_model,
                    inpaint_model,
                    sobel_conv
                )
                result['index'] = i
                result['filename'] = image_file.filename
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'filename': image_file.filename,
                    'success': False,
                    'error': str(e)
                })

        total_end_time = time.time()
        total_time_ms = int((total_end_time - total_start_time) * 1000)

        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(images_files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_time_ms': total_time_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_batch', methods=['POST'])
def download_batch():
    """
    将批量处理结果打包为ZIP文件下载
    """
    try:
        data = request.json
        if 'results' not in data:
            return jsonify({'error': '缺少结果数据'}), 400

        results = data['results']
        
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, result in enumerate(results):
                if not result.get('success'):
                    continue
                
                filename_base = result.get('filename', f'image_{idx}').rsplit('.', 1)[0]
                
                repaired_data = result['repaired_image'].split(',')[1]
                repaired_bytes = base64.b64decode(repaired_data)
                zf.writestr(f'{filename_base}_repaired.png', repaired_bytes)
                
                edge_data = result['edge_image'].split(',')[1]
                edge_bytes = base64.b64decode(edge_data)
                zf.writestr(f'{filename_base}_edge.png', edge_bytes)

        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='repaired_images.zip'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('frontend', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
