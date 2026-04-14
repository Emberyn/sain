const uploadInput = document.getElementById('uploadInput');
const maskUploadInput = document.getElementById('maskUploadInput');
const uploadZone = document.getElementById('uploadZone');
const thumbnailSection = document.getElementById('thumbnailSection');
const thumbnailGrid = document.getElementById('thumbnailGrid');
const editorSection = document.getElementById('editorSection');
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const previewCanvas = document.getElementById('previewCanvas');
const imgCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const previewCtx = previewCanvas.getContext('2d');
const brushSizeInput = document.getElementById('brushSize');
const statusPanel = document.getElementById('statusPanel');
const repairBtn = document.getElementById('repairBtn');
const downloadBtn = document.getElementById('downloadBtn');
const progressBar = document.getElementById('progressBar');
const resultsSection = document.getElementById('resultsSection');
const repairedGrid = document.getElementById('repairedGrid');
const edgeGrid = document.getElementById('edgeGrid');
const originalGrid = document.getElementById('originalGrid');
const tabRepaired = document.getElementById('tabRepaired');
const tabEdge = document.getElementById('tabEdge');
const tabOriginal = document.getElementById('tabOriginal');
const currentImageName = document.getElementById('currentImageName');
const imageCount = document.getElementById('imageCount');

// Layout zoom slider elements
const layoutZoomSlider = document.getElementById('layoutZoom');
const layoutZoomValue = document.getElementById('layoutZoomValue');

// Result page layout zoom slider elements
const resultLayoutZoomSlider = document.getElementById('resultLayoutZoom');
const resultLayoutZoomValue = document.getElementById('resultLayoutZoomValue');

// Modal elements
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalTitle = document.getElementById('modalTitle');
const modalTabRepaired = document.getElementById('modalTabRepaired');
const modalTabEdge = document.getElementById('modalTabEdge');
const modalTabOriginal = document.getElementById('modalTabOriginal');

let currentModalIndex = -1; // Current image index in modal
let currentModalView = 'repaired'; // Current view mode in modal

let uploadedImages = [];
let currentImageIndex = -1;
let isDrawing = false;
let batchResults = [];

// 画笔位置变量
let brushX = 128; // 初始X位置（画布中心）
let brushY = 128; // 初始Y位置（画布中心）
const stepSize = 5; // 每次移动的步长

// 图形工具相关
let currentTool = 'brush'; // 当前工具: brush, eraser, hand, circle, ellipse, rect, triangle
let isShapeDrawing = false; // 是否正在绘制形状
let shapeStartX = 0;
let shapeStartY = 0;

// 缩放和平移相关
let scale = 1; // 当前缩放比例
let translateX = 0; // X轴平移
let translateY = 0; // Y轴平移
let isPanning = false; // 是否正在平移
let panStartX = 0; // 平移起始X
let panStartY = 0; // 平移起始Y

maskCtx.fillStyle = 'rgba(0,0,0,0)';
maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

// 初始状态下隐藏画笔大小控制和结果页面布局调节（只在相应页面显示）
document.getElementById('brushSizeControl').style.display = 'none';
document.getElementById('resultZoomControl').style.display = 'none';

// Layout zoom slider event listener - 只控制缩略图grid
layoutZoomSlider.addEventListener('input', function() {
    const size = this.value;
    layoutZoomValue.textContent = size + 'px';
    const gridStyle = `repeat(auto-fill, minmax(${size}px, 1fr))`;
    thumbnailGrid.style.gridTemplateColumns = gridStyle;
});

// Result page layout zoom slider event listener
resultLayoutZoomSlider.addEventListener('input', function() {
    const size = this.value;
    resultLayoutZoomValue.textContent = size + 'px';
    const gridStyle = `repeat(auto-fill, minmax(${size}px, 1fr))`;
    repairedGrid.style.gridTemplateColumns = gridStyle;
    edgeGrid.style.gridTemplateColumns = gridStyle;
    originalGrid.style.gridTemplateColumns = gridStyle;
});

uploadInput.addEventListener('change', handleFiles);
maskUploadInput.addEventListener('change', handleMaskFiles);

// 为编辑器容器添加滚轮和平移事件
const editorContainer = document.getElementById('editorContainer');
editorContainer.addEventListener('wheel', handleWheel, { passive: false });
editorContainer.addEventListener('mousedown', startPan);
editorContainer.addEventListener('mousemove', pan);
editorContainer.addEventListener('mouseup', endPan);
editorContainer.addEventListener('mouseleave', endPan);

function handleUploadClick() {
    // 只有在修复完成后，才清除所有数据
    // batchResults.length > 0 表示已经完成过修复
    if (batchResults.length > 0) {
        clearAllData();
    }
    // 触发文件选择
    document.getElementById('uploadInput').click();
}

function handleMaskUploadClick() {
    // 触发掩码文件选择
    document.getElementById('maskUploadInput').click();
}

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    handleFileList(files);
});

function handleFiles(e) {
    const files = Array.from(e.target.files);
    handleFileList(files);
}

function handleMaskFiles(e) {
    const files = Array.from(e.target.files);
    handleMaskFileList(files);
}

function handleFileList(files) {
    if (uploadedImages.length + files.length > 20) {
        alert('最多只能上传20张图片！');
        return;
    }

    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                uploadedImages.push({
                    file: file,
                    src: event.target.result,
                    image: img,
                    mask: null,
                    completed: false,
                    maskFile: null  // 用于存储掩码文件
                });
                updateThumbnailGrid();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function handleMaskFileList(files) {
    if (uploadedImages.length === 0) {
        alert('请先上传图片！');
        return;
    }

    if (files.length !== uploadedImages.length) {
        alert(`掩码文件数量(${files.length})与图片数量(${uploadedImages.length})不匹配！`);
        return;
    }

    // 按文件名排序以确保正确配对
    const sortedFiles = files.sort((a, b) => a.name.localeCompare(b.name));
    const sortedImages = [...uploadedImages].sort((a, b) => a.file.name.localeCompare(b.file.name));

    let processedCount = 0;
    
    sortedFiles.forEach((maskFile, index) => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const maskImg = new Image();
            maskImg.onload = function() {
                // 创建临时canvas来处理掩码
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 256;
                tempCanvas.height = 256;
                const tempCtx = tempCanvas.getContext('2d');

                // 绘制掩码图像
                tempCtx.drawImage(maskImg, 0, 0, 256, 256);
                const maskData = tempCtx.getImageData(0, 0, 256, 256);

                // 找到对应的图片并设置掩码
                const originalIndex = uploadedImages.findIndex(img => img.file.name === sortedImages[index].file.name);
                if (originalIndex !== -1) {
                    uploadedImages[originalIndex].mask = maskData;
                    uploadedImages[originalIndex].completed = true;
                    uploadedImages[originalIndex].maskFile = maskFile;
                }

                processedCount++;
                if (processedCount === files.length) {
                    updateThumbnailGrid();
                    statusPanel.innerHTML = `<strong>系统状态：</strong> 已上传 ${files.length} 个掩码文件，所有图片已完成！点击"一键智能修复"开始处理。`;
                }
            };
            maskImg.src = event.target.result;
        };
        reader.readAsDataURL(maskFile);
    });
}

function updateThumbnailGrid() {
    thumbnailGrid.innerHTML = '';
    imageCount.textContent = uploadedImages.length;

    if (uploadedImages.length > 0) {
        thumbnailSection.classList.remove('hidden');
        uploadZone.style.display = 'none';
        // 显示掩码上传按钮
        document.getElementById('maskUploadBtn').style.display = 'inline-flex';
    }

    uploadedImages.forEach((item, index) => {
        const div = document.createElement('div');
        // 如果有掩码文件，也标记为completed
        const isCompleted = item.completed || item.maskFile;
        div.className = 'thumbnail-item' + (isCompleted ? ' completed' : '') + (index === currentImageIndex ? ' active' : '');
        
        // 删除按钮
        const deleteBtn = document.createElement('div');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation(); // 阻止触发缩略图点击事件
            deleteImage(index);
        };
        div.appendChild(deleteBtn);
        
        // 图片
        div.onclick = () => openEditor(index);
        
        const img = document.createElement('img');
        
        // 如果有遮罩，合成遮罩后的图片
        if (item.mask) {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = item.image.width;
            tempCanvas.height = item.image.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // 绘制原始图片
            tempCtx.drawImage(item.image, 0, 0);
            
            // 将遮罩数据绘制到图片上（半透明红色）
            const maskData = item.mask;
            for (let y = 0; y < maskData.height; y++) {
                for (let x = 0; x < maskData.width; x++) {
                    const maskIndex = (y * maskData.width + x) * 4;
                    if (maskData.data[maskIndex] > 0) { // 如果遮罩区域不为透明
                        const imgIndex = (y * item.image.width + x) * 4;
                        // 绘制半透明红色遮罩
                        tempCtx.fillStyle = 'rgba(255, 0, 0, 0.4)';
                        tempCtx.fillRect(x, y, 1, 1);
                    }
                }
            }
            
            img.src = tempCanvas.toDataURL();
        } else {
            img.src = item.src;
        }
        
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = item.file.name.substring(0, 15) + (item.file.name.length > 15 ? '...' : '');
        
        div.appendChild(img);
        div.appendChild(label);
        thumbnailGrid.appendChild(div);
    });

    checkCanRepair();
}

function deleteImage(index) {
    // 确认删除
    const itemName = uploadedImages[index].file.name;
    if (!confirm(`确定要删除图片 "${itemName}" 吗？`)) {
        return;
    }
    
    uploadedImages.splice(index, 1);
    
    // 如果当前正在编辑这张图片，关闭编辑器
    if (currentImageIndex === index) {
        currentImageIndex = -1;
        editorSection.classList.add('hidden');
        thumbnailSection.classList.remove('hidden');
    } else if (currentImageIndex > index) {
        // 调整当前编辑索引
        currentImageIndex--;
    }
    
    updateThumbnailGrid();
    
    if (uploadedImages.length === 0) {
        thumbnailSection.classList.add('hidden');
        uploadZone.style.display = 'flex';
        statusPanel.innerHTML = '<strong>系统状态：</strong> 已删除图片。请重新上传。';
    } else {
        statusPanel.innerHTML = `<strong>系统状态：</strong> 已删除图片，当前剩余 ${uploadedImages.length} 张图片。`;
    }
    
    // 清空文件输入框，允许重新上传相同文件
    uploadInput.value = '';
}

function openEditor(index) {
    currentImageIndex = index;
    const item = uploadedImages[index];
                
    // 重置画笔位置到画布中心
    brushX = imageCanvas.width / 2;
    brushY = imageCanvas.height / 2;
            
    // 重置缩放和平移
    scale = 1;
    translateX = 0;
    translateY = 0;
    updateCanvasTransform();
                
    currentImageName.textContent = item.file.name;
                
    imgCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    imgCtx.drawImage(item.image, 0, 0, imageCanvas.width, imageCanvas.height);
                
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (item.mask) {
        maskCtx.putImageData(item.mask, 0, 0);
    } else {
        maskCtx.fillStyle = 'rgba(0,0,0,0)';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
                
    thumbnailSection.classList.add('hidden');
    editorSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
        
    // 隐藏导航栏上的上传按钮、上传区域和调节布局，显示画笔大小
    document.getElementById('uploadBtn').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('zoomControl').style.display = 'none';
    document.getElementById('brushSizeControl').style.display = 'block';
        
    // 隐藏编辑页面中不需要的面板
    document.getElementById('edgePanel').style.display = 'none';
    document.getElementById('repairPanel').style.display = 'none';
                
    statusPanel.innerHTML = '<strong>系统状态：</strong> 请在图片上涂抹需要修复的病斑或破损区域。使用"手型"工具可以缩放和平移图片，方便绘制细节。';
}

function saveCurrentMask() {
    if (currentImageIndex >= 0) {
        uploadedImages[currentImageIndex].mask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        uploadedImages[currentImageIndex].completed = true;
        updateThumbnailGrid();
    }
            
    editorSection.classList.add('hidden');
    thumbnailSection.classList.remove('hidden');
    
    // 隐藏画笔大小，恢复上传按钮、上传区域和调节布局
    document.getElementById('brushSizeControl').style.display = 'none';
    document.getElementById('uploadBtn').style.display = 'block';
    if (uploadedImages.length === 0) {
        document.getElementById('uploadZone').style.display = 'flex';
    }
    document.getElementById('zoomControl').style.display = 'flex';
    
    // 恢复编辑页面中隐藏的面板
    document.getElementById('edgePanel').style.display = 'block';
    document.getElementById('repairPanel').style.display = 'block';
            
    const remaining = uploadedImages.filter(img => !img.completed).length;
    if (remaining === 0) {
        statusPanel.innerHTML = '<strong>系统状态：</strong> 所有图片遮罩已完成！点击"一键智能修复"开始处理。';
    } else {
        statusPanel.innerHTML = `<strong>系统状态：</strong> 还有 ${remaining} 张图片需要绘制遮罩。`;
    }
}

function clearMask() {
    // 清除当前遮罩
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.fillStyle = 'rgba(0,0,0,0)';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    statusPanel.innerHTML = '<strong>系统状态：</strong> 遮罩已清除，请重新绘制。';
}

function checkCanRepair() {
    // 检查所有图片是否都已完成（手动绘制或上传掩码）
    const allCompleted = uploadedImages.length > 0 && uploadedImages.every(img => img.completed || img.maskFile);
    repairBtn.disabled = !allCompleted;
}

function clearAllData() {
    // 清除所有数据但不显示确认对话框
    uploadedImages = [];
    currentImageIndex = -1;
    batchResults = [];
    
    thumbnailGrid.innerHTML = '';
    thumbnailSection.classList.add('hidden');
    editorSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    uploadZone.style.display = 'block';
    
    repairBtn.disabled = true;
    downloadBtn.style.display = 'none';
    progressBar.style.display = 'none';
    document.getElementById('maskUploadBtn').style.display = 'none';
    document.getElementById('resultZoomControl').style.display = 'none';
    
    imgCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    document.getElementById('repairResult').src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
    document.getElementById('edgeResult').src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
    
    statusPanel.innerHTML = '<strong>系统状态：</strong> 已清除所有内容。请重新上传图片。';
}

function clearAll() {
    if (confirm('确定要清除所有图片和遮罩吗？')) {
        clearAllData();
    }
}

maskCanvas.addEventListener('mousedown', function(e) {
    if (currentImageIndex < 0) return;
    
    // 手型模式下不处理绘制
    if (currentTool === 'hand') return;
    
    if (currentTool === 'brush' || currentTool === 'eraser') {
        // 画笔和橡皮擦工具：拖拽时连续绘制
        isDrawing = true;
        const pos = getMousePos(maskCanvas, e);
        brushX = pos.x;
        brushY = pos.y;
        draw(e);
    } else {
        // 图形工具：鼠标拖拽绘制形状
        isShapeDrawing = true;
        const pos = getMousePos(maskCanvas, e);
        shapeStartX = pos.x;
        shapeStartY = pos.y;
    }
});

maskCanvas.addEventListener('mousemove', function(e) {
    if (currentImageIndex < 0) return;
    
    // 手型模式下不处理绘制
    if (currentTool === 'hand') return;
    
    const pos = getMousePos(maskCanvas, e);
    
    // 始终更新光标显示（画笔和橡皮擦）
    if (currentTool === 'brush' || currentTool === 'eraser') {
        updateCursor(pos.x, pos.y);
    }
    
    if (currentTool === 'brush' || currentTool === 'eraser') {
        // 画笔和橡皮擦：拖拽时连续绘制
        if (!isDrawing) return;
        draw(e);
    } else if (isShapeDrawing) {
        // 图形工具：拖拽时预览形状
        previewShapeOnCanvas(shapeStartX, shapeStartY, pos.x, pos.y);
    }
});

maskCanvas.addEventListener('mouseup', function(e) {
    if (currentTool !== 'brush' && isShapeDrawing) {
        // 图形工具：释放鼠标时完成形状绘制
        const pos = getMousePos(maskCanvas, e);
        drawShapeByDrag(shapeStartX, shapeStartY, pos.x, pos.y);
        isShapeDrawing = false;
        // 清除预览画布
        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
    isDrawing = false;
});

maskCanvas.addEventListener('mouseout', function(e) {
    if (currentTool !== 'brush' && isShapeDrawing) {
        // 鼠标离开画布时完成形状绘制
        const pos = getMousePos(maskCanvas, e);
        drawShapeByDrag(shapeStartX, shapeStartY, pos.x, pos.y);
        isShapeDrawing = false;
        // 清除预览画布
        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    }
    isDrawing = false;
});

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

function draw(e) {
    const pos = getMousePos(maskCanvas, e);
    brushX = pos.x;
    brushY = pos.y;
    
    // 获取基础画笔大小，并根据缩放比例调整
    const baseSize = parseInt(brushSizeInput.value);
    // 画笔大小应该与缩放成反比：放大时画笔覆盖更少像素
    const size = baseSize / scale;
    
    if (currentTool === 'eraser') {
        // 橡皮擦：使用 destination-out 混合模式擦除
        maskCtx.save();
        maskCtx.globalCompositeOperation = 'destination-out';
        maskCtx.beginPath();
        maskCtx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
        maskCtx.fill();
        maskCtx.restore();
    } else {
        // 画笔：绘制红色遮罩
        maskCtx.fillStyle = '#ff0000';
        maskCtx.beginPath();
        maskCtx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
        maskCtx.fill();
    }
}

// 橡皮擦：直接擦除像素
function eraseAt(cx, cy, radius) {
    const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;
    const radiusSq = radius * radius;
    
    // 计算擦除范围（扩大一点以确保覆盖整个圆）
    const startX = Math.max(0, Math.floor(cx - radius - 1));
    const endX = Math.min(maskCanvas.width - 1, Math.ceil(cx + radius + 1));
    const startY = Math.max(0, Math.floor(cy - radius - 1));
    const endY = Math.min(maskCanvas.height - 1, Math.ceil(cy + radius + 1));
    
    // 将范围内的像素设为完全透明
    for (let y = startY; y <= endY; y++) {
        for (let x = startX; x <= endX; x++) {
            const dx = x - cx;
            const dy = y - cy;
            if (dx * dx + dy * dy <= radiusSq) {
                const idx = (y * maskCanvas.width + x) * 4;
                // 将RGBA全部设为0，完全擦除
                data[idx] = 0;     // R
                data[idx + 1] = 0; // G
                data[idx + 2] = 0; // B
                data[idx + 3] = 0; // Alpha 设为 0（透明）
            }
        }
    }
    
    maskCtx.putImageData(imageData, 0, 0);
}

// 更新光标位置显示
function updateCursor(x, y) {
    brushX = x;
    brushY = y;
    
    if (currentTool === 'brush' || currentTool === 'eraser') {
        // 清除之前的光标
        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
        
        // 获取基础画笔大小，并根据缩放比例调整
        const baseSize = parseInt(brushSizeInput.value);
        const size = baseSize / scale;
        
        // 绘制深色实心圆表示画笔/橡皮擦大小，增强可见性
        const isEraser = currentTool === 'eraser';
        // 橡皮擦使用红色半透明填充，画笔使用深灰色
        previewCtx.fillStyle = isEraser ? 'rgba(255, 80, 80, 0.5)' : 'rgba(40, 40, 40, 0.6)';
        previewCtx.strokeStyle = isEraser ? 'rgba(220, 0, 0, 1.0)' : 'rgba(20, 20, 20, 1.0)';
        previewCtx.lineWidth = 2;
        previewCtx.beginPath();
        previewCtx.arc(x, y, size, 0, Math.PI * 2);
        previewCtx.fill();
        previewCtx.stroke();
    }
}

// 设置当前工具
function setTool(tool) {
    currentTool = tool;
    
    // 更新按钮状态
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(tool + 'Tool').classList.add('active');
    
    // 隐藏形状参数（现在用拖拽控制）
    const shapeParams = document.getElementById('shapeParams');
    shapeParams.style.display = 'none';
    
    // 更新画笔大小标签
    const brushSizeLabel = document.querySelector('label[for="brushSize"]');
    
    // 清除之前的光标
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // 更新容器样式
    const container = document.getElementById('editorContainer');
    if (tool === 'hand') {
        container.classList.add('zoomed');
    } else {
        container.classList.remove('zoomed');
    }
    
    // 更新状态提示
    if (tool === 'brush') {
        statusPanel.innerHTML = '<strong>系统状态：</strong> 画笔模式 - 按住鼠标拖动或使用方向键涂抹病斑区域。';
        brushSizeLabel.textContent = '画笔大小:';
    } else if (tool === 'eraser') {
        statusPanel.innerHTML = '<strong>系统状态：</strong> 橡皮擦模式 - 按住鼠标拖动或使用方向键擦除遮罩。';
        brushSizeLabel.textContent = '橡皮擦大小:';
    } else if (tool === 'hand') {
        statusPanel.innerHTML = '<strong>系统状态：</strong> 拖动放大模式 - 滚动鼠标滚轮缩放，按住鼠标拖动平移图片。';
        brushSizeLabel.textContent = '画笔大小:';
    } else {
        const toolNames = {circle: '圆形', ellipse: '椭圆', rect: '矩形', triangle: '三角形'};
        statusPanel.innerHTML = `<strong>系统状态：</strong> ${toolNames[tool]}模式 - 按住鼠标拖拽绘制形状，像PPT一样控制大小和位置。`;
        brushSizeLabel.textContent = '画笔大小:';
    }
}

// 更新画布变换（缩放和平移）
function updateCanvasTransform() {
    const canvases = [imageCanvas, maskCanvas, previewCanvas];
    canvases.forEach(canvas => {
        canvas.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
        canvas.style.transformOrigin = 'center center';
    });
}

// 处理鼠标滚轮缩放
function handleWheel(e) {
    if (currentTool !== 'hand' || currentImageIndex < 0) return;
    e.preventDefault();
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1; // 向下滚动缩小，向上滚动放大
    const newScale = Math.min(Math.max(scale * delta, 0.5), 5); // 限制缩放范围0.5x-5x
    
    scale = newScale;
    updateCanvasTransform();
    
    statusPanel.innerHTML = `<strong>系统状态：</strong> 缩放比例: ${(scale * 100).toFixed(0)}%`;
}

// 开始平移
function startPan(e) {
    if (currentTool !== 'hand' || currentImageIndex < 0) return;
    isPanning = true;
    panStartX = e.clientX - translateX;
    panStartY = e.clientY - translateY;
    document.getElementById('editorContainer').style.cursor = 'grabbing';
}

// 平移中
function pan(e) {
    if (!isPanning || currentTool !== 'hand') return;
    e.preventDefault();
    
    translateX = e.clientX - panStartX;
    translateY = e.clientY - panStartY;
    updateCanvasTransform();
}

// 结束平移
function endPan() {
    isPanning = false;
    if (currentTool === 'hand') {
        document.getElementById('editorContainer').style.cursor = 'grab';
    }
}

// 预览形状（在预览画布上显示）
function previewShapeOnCanvas(x1, y1, x2, y2) {
    // 清除之前的预览
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // 绘制半透明预览形状
    previewCtx.fillStyle = 'rgba(255, 0, 0, 0.4)';
    previewCtx.strokeStyle = '#ff0000';
    previewCtx.lineWidth = 2;
    
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    
    if (width < 3 || height < 3) return;
    
    previewCtx.beginPath();
    
    switch(currentTool) {
        case 'circle':
            const radius = Math.max(width, height) / 2;
            previewCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            break;
        
        case 'ellipse':
            previewCtx.ellipse(centerX, centerY, width / 2, height / 2, 0, 0, Math.PI * 2);
            break;
        
        case 'rect':
            previewCtx.rect(Math.min(x1, x2), Math.min(y1, y2), width, height);
            break;
        
        case 'triangle':
            // 绘制三角形：顶点在上，底部在下
            const topX = centerX;
            const topY = Math.min(y1, y2);
            const bottomLeftX = Math.min(x1, x2);
            const bottomRightX = Math.max(x1, x2);
            const bottomY = Math.max(y1, y2);
            previewCtx.moveTo(topX, topY);
            previewCtx.lineTo(bottomLeftX, bottomY);
            previewCtx.lineTo(bottomRightX, bottomY);
            previewCtx.closePath();
            break;
    }
    
    previewCtx.fill();
    previewCtx.stroke();
}

// 根据拖拽坐标绘制形状
function drawShapeByDrag(x1, y1, x2, y2) {
    maskCtx.fillStyle = '#ff0000';
    drawShapeAtCoordinates(x1, y1, x2, y2);
}

// 在指定坐标绘制形状
function drawShapeAtCoordinates(x1, y1, x2, y2) {
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    
    // 确保形状有最小尺寸
    if (width < 5 || height < 5) return;
    
    maskCtx.beginPath();
    
    switch(currentTool) {
        case 'circle':
            // 圆形：使用宽高中的较大值作为直径
            const radius = Math.max(width, height) / 2;
            maskCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            break;
        
        case 'ellipse':
            // 椭圆：使用拖拽的宽高
            maskCtx.ellipse(centerX, centerY, width / 2, height / 2, 0, 0, Math.PI * 2);
            break;
        
        case 'rect':
            // 矩形：使用拖拽的起始点和终点
            maskCtx.rect(Math.min(x1, x2), Math.min(y1, y2), width, height);
            break;
        
        case 'triangle':
            // 绘制三角形：顶点在上，底部在下
            const topX = centerX;
            const topY = Math.min(y1, y2);
            const bottomLeftX = Math.min(x1, x2);
            const bottomRightX = Math.max(x1, x2);
            const bottomY = Math.max(y1, y2);
            maskCtx.moveTo(topX, topY);
            maskCtx.lineTo(bottomLeftX, bottomY);
            maskCtx.lineTo(bottomRightX, bottomY);
            maskCtx.closePath();
            break;
    }
    
    maskCtx.fill();
}

// 原有的按参数绘制形状（保留以便键盘操作）
function drawShape(x, y) {
    maskCtx.fillStyle = '#ff0000';
    const width = parseInt(document.getElementById('shapeWidth').value);
    const height = parseInt(document.getElementById('shapeHeight').value);
    
    maskCtx.beginPath();
    
    switch(currentTool) {
        case 'circle':
            const radius = width / 2;
            maskCtx.arc(x, y, radius, 0, Math.PI * 2);
            break;
        
        case 'ellipse':
            maskCtx.ellipse(x, y, width / 2, height / 2, 0, 0, Math.PI * 2);
            break;
        
        case 'rect':
            maskCtx.rect(x - width / 2, y - height / 2, width, height);
            break;
    }
    
    maskCtx.fill();
}

// 绘制画笔光标
function drawBrushCursor() {
    if (currentImageIndex < 0) return;
    
    // 创建临时画布来显示光标
    const cursorCanvas = document.createElement('canvas');
    cursorCanvas.width = maskCanvas.width;
    cursorCanvas.height = maskCanvas.height;
    const cursorCtx = cursorCanvas.getContext('2d');
    
    // 复制当前遮罩内容
    cursorCtx.putImageData(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height), 0, 0);
    
    // 绘制画笔光标
    cursorCtx.strokeStyle = '#ff0000';
    cursorCtx.lineWidth = 1;
    cursorCtx.beginPath();
    cursorCtx.arc(brushX, brushY, parseInt(brushSizeInput.value), 0, Math.PI * 2);
    cursorCtx.stroke();
    
    // 更新显示
    maskCanvas.style.backgroundImage = `url(${cursorCanvas.toDataURL()})`;
}

// 键盘控制画笔移动
document.addEventListener('keydown', function(e) {
    if (currentImageIndex < 0 || editorSection.classList.contains('hidden')) return;
    
    let moved = false;
    switch(e.key) {
        case 'ArrowUp':
            brushY = Math.max(0, brushY - stepSize);
            moved = true;
            break;
        case 'ArrowDown':
            brushY = Math.min(maskCanvas.height, brushY + stepSize);
            moved = true;
            break;
        case 'ArrowLeft':
            brushX = Math.max(0, brushX - stepSize);
            moved = true;
            break;
        case 'ArrowRight':
            brushX = Math.min(maskCanvas.width, brushX + stepSize);
            moved = true;
            break;
        case 'Enter':
            // 回车键放置形状
            if (currentTool !== 'brush' && currentTool !== 'eraser') {
                e.preventDefault();
                drawShape(brushX, brushY);
                return;
            }
            break;
    }
    
    if (moved) {
        e.preventDefault();
        // 更新光标显示
        updateCursor(brushX, brushY);
        
        // 获取基础画笔大小，并根据缩放比例调整
        const baseSize = parseInt(brushSizeInput.value);
        const size = baseSize / scale;
        
        // 根据当前工具类型处理
        if (currentTool === 'brush') {
            // 画笔模式：移动时直接绘制
            maskCtx.fillStyle = '#ff0000';
            maskCtx.beginPath();
            maskCtx.arc(brushX, brushY, size, 0, Math.PI * 2);
            maskCtx.fill();
        } else if (currentTool === 'eraser') {
            // 橡皮擦模式：移动时擦除
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(brushX, brushY, size, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.restore();
        }
    }
});

async function submitBatchRepair() {
    if (uploadedImages.length === 0) {
        alert('请先上传图片！');
        return;
    }

    repairBtn.disabled = true;
    downloadBtn.style.display = 'none';
    progressBar.style.display = 'block';
    
    // 根据图片数量显示不同的提示
    const imageCount = uploadedImages.length;
    const processText = imageCount === 1 ? '正在处理' : `正在批量处理 ${imageCount} 张`;
    statusPanel.innerHTML = `<strong>系统状态：</strong> ${processText}图片，请耐心等待...`;

    try {
        const formData = new FormData();
        
        // 使用Promise等待所有blob创建完成
        const blobPromises = [];
        
        uploadedImages.forEach((item, index) => {
            formData.append('images', item.file);
            
            // 如果有上传的掩码文件，直接使用
            if (item.maskFile) {
                formData.append('masks', item.maskFile);
                blobPromises.push(Promise.resolve());
            } else {
                // 否则从 canvas 创建掩码
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskCanvas.width;
                tempCanvas.height = maskCanvas.height;
                const tCtx = tempCanvas.getContext('2d');
                
                tCtx.fillStyle = '#000000';
                tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                
                const maskData = item.mask;
                const tData = tCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                
                for (let i = 0; i < maskData.data.length; i += 4) {
                    if (maskData.data[i] > 0) {
                        tData.data[i] = 255;
                        tData.data[i+1] = 255;
                        tData.data[i+2] = 255;
                    }
                }
                tCtx.putImageData(tData, 0, 0);
                
                // 创建一个Promise来等待blob创建完成
                const blobPromise = new Promise((resolve) => {
                    tempCanvas.toBlob((blob) => {
                        formData.append('masks', blob, `mask_${index}.png`);
                        resolve();
                    }, 'image/png');
                });
                blobPromises.push(blobPromise);
            }
        });

        // 等待所有blob创建完成
        await Promise.all(blobPromises);

        // 根据图片数量显示不同的提示
        const processText = imageCount === 1 ? '正在处理' : `正在批量处理 ${imageCount} 张`;
        statusPanel.innerHTML = `<strong>系统状态：</strong> ${processText}图片，请稍候...`;

        // 设置请求超时为5分钟（处理20张图片需要更长时间）
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5分钟超时

        const response = await fetch('/api/repair_batch', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        const data = await response.json();

        processResults(data);
        repairBtn.disabled = false;
        progressBar.style.display = 'none';
    } catch (error) {
        console.error(error);
        if (error.name === 'AbortError') {
            alert('处理超时！图片数量较多，请稍后重试或分批处理。');
            statusPanel.innerHTML = '<strong>系统状态：</strong> 处理超时，请重试。';
        } else {
            alert('网络或服务器错误！');
            statusPanel.innerHTML = '<strong>系统状态：</strong> 网络或服务器错误。';
        }
        repairBtn.disabled = false;
        progressBar.style.display = 'none';
    }
}

// 处理修复结果的函数
function processResults(data) {
    if (data.success) {
        batchResults = data.results;
        displayBatchResults(data);
        downloadBtn.style.display = 'inline-flex';
        document.getElementById('reuploadBtn').style.display = 'inline-flex';
        
        // 根据图片数量显示不同的提示
        const imageCount = uploadedImages.length;
        const completeText = imageCount === 1 ? '修复完成！' : `批量修复完成！`;
        statusPanel.innerHTML = `<strong>${completeText}</strong> 成功: ${data.successful}/${data.total_images}, 失败: ${data.failed}, 总耗时: ${data.total_time_ms} ms`;
    } else {
        alert('修复失败：' + data.error);
        statusPanel.innerHTML = '<strong>系统状态：</strong> 修复失败，请查看控制台日志。';
    }
}

function displayBatchResults(data) {
    repairedGrid.innerHTML = '';
    edgeGrid.innerHTML = '';
    originalGrid.innerHTML = '';
    resultsSection.classList.remove('hidden');
    
    // 隐藏导航栏上不需要的功能
    document.getElementById('uploadBtn').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('brushSizeControl').style.display = 'none';
    document.getElementById('zoomControl').style.display = 'none';
    document.getElementById('repairBtn').style.display = 'none';
    document.getElementById('maskUploadBtn').style.display = 'none';
    
    // 显示结果页面的布局调节滑动条
    document.getElementById('resultZoomControl').style.display = 'flex';
    
    // 应用结果页面布局滑动条的初始值
    const initialSize = resultLayoutZoomSlider.value;
    resultLayoutZoomValue.textContent = initialSize + 'px';
    const gridStyle = `repeat(auto-fill, minmax(${initialSize}px, 1fr))`;
    repairedGrid.style.gridTemplateColumns = gridStyle;
    edgeGrid.style.gridTemplateColumns = gridStyle;
    originalGrid.style.gridTemplateColumns = gridStyle;
    
    // 显示修复后的图片
    data.results.forEach((result, idx) => {
        if (!result.success) {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <img src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="处理失败">
                <div class="card-title">${result.filename}</div>
                <p style="color: red; padding: 5px; margin: 0;">处理失败: ${result.error}</p>
            `;
            repairedGrid.appendChild(card);
            return;
        }

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <img src="${result.repaired_image}" alt="修复结果" data-index="${idx}" ondblclick="openModal(${idx})">
            <div class="card-title">${result.filename}</div>
            <div style="font-size: 12px; color: #666; padding: 5px;">耗时: ${result.inference_time_ms} ms</div>
            <button class="btn" style="margin: 5px; padding: 5px 10px; font-size: 14px; width: calc(100% - 10px);" onclick="downloadSingle(${idx})">下载此结果</button>
        `;
        repairedGrid.appendChild(card);
    });
    
    // 显示 SAIN 结构感知叶脉图
    data.results.forEach((result, idx) => {
        if (!result.success) {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <img src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="处理失败">
                <div class="card-title">${result.filename}</div>
                <p style="color: red; padding: 5px; margin: 0;">处理失败: ${result.error}</p>
            `;
            edgeGrid.appendChild(card);
            return;
        }

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <img src="${result.edge_image}" alt="SAIN 结构感知叶脉图" data-index="${idx}" ondblclick="openModal(${idx})">
            <div class="card-title">${result.filename}</div>
            <div style="font-size: 12px; color: #666; padding: 5px;">叶脉边缘图</div>
            <button class="btn" style="margin: 5px; padding: 5px 10px; font-size: 14px; width: calc(100% - 10px);" onclick="downloadEdgeImage(${idx})">下载叶脉图</button>
        `;
        edgeGrid.appendChild(card);
    });
    
    // 显示修复前的图片（原始上传的图片）
    uploadedImages.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <img src="${item.src}" alt="原始图片" data-index="${idx}" ondblclick="openModal(${idx})">
            <div class="card-title">${item.file.name}</div>
            <div style="font-size: 12px; color: #666; padding: 5px;">原始图片</div>
        `;
        originalGrid.appendChild(card);
    });

    editorSection.classList.add('hidden');
    thumbnailSection.classList.add('hidden');
    
    // 默认显示修复后标签
    switchTab('repaired');
}

function switchTab(tab) {
    if (tab === 'repaired') {
        repairedGrid.classList.remove('hidden');
        edgeGrid.classList.add('hidden');
        originalGrid.classList.add('hidden');
        tabRepaired.style.backgroundColor = 'var(--primary-color)';
        tabEdge.style.backgroundColor = '#999';
        tabOriginal.style.backgroundColor = '#999';
    } else if (tab === 'edge') {
        repairedGrid.classList.add('hidden');
        edgeGrid.classList.remove('hidden');
        originalGrid.classList.add('hidden');
        tabRepaired.style.backgroundColor = '#999';
        tabEdge.style.backgroundColor = 'var(--primary-color)';
        tabOriginal.style.backgroundColor = '#999';
    } else {
        repairedGrid.classList.add('hidden');
        edgeGrid.classList.add('hidden');
        originalGrid.classList.remove('hidden');
        tabRepaired.style.backgroundColor = '#999';
        tabEdge.style.backgroundColor = '#999';
        tabOriginal.style.backgroundColor = 'var(--primary-color)';
    }
}

// 全局键盘导航（修复结果页面和模态框）
document.addEventListener('keydown', function(e) {
    const isModalOpen = imageModal.classList.contains('show');
    const isResultsVisible = !resultsSection.classList.contains('hidden');
    
    // 如果两个都不可见，不处理键盘事件
    if (!isModalOpen && !isResultsVisible) {
        return;
    }
    
    // ESC 键关闭模态框
    if (e.key === 'Escape' && isModalOpen) {
        e.preventDefault();
        closeModal();
        return;
    }
    
    // 左方向键
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (isModalOpen) {
            // 模态框中的视图切换：修复后 -> 修复前 -> 叶脉图 -> 修复后
            if (currentModalView === 'repaired') {
                switchModalView('original');
            } else if (currentModalView === 'original') {
                switchModalView('edge');
            } else {
                switchModalView('repaired');
            }
        } else if (isResultsVisible) {
            // 结果页面的视图切换：修复后 -> 修复前 -> 叶脉图 -> 修复后
            let currentTab = 'repaired';
            if (!edgeGrid.classList.contains('hidden')) {
                currentTab = 'edge';
            } else if (!originalGrid.classList.contains('hidden')) {
                currentTab = 'original';
            }
            
            const tabOrder = ['repaired', 'original', 'edge'];
            const currentIndex = tabOrder.indexOf(currentTab);
            const prevIndex = (currentIndex + 1) % tabOrder.length;
            switchTab(tabOrder[prevIndex]);
        }
    }
    // 右方向键
    else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (isModalOpen) {
            // 模态框中的视图切换：修复后 -> 叶脉图 -> 修复前 -> 修复后
            if (currentModalView === 'repaired') {
                switchModalView('edge');
            } else if (currentModalView === 'edge') {
                switchModalView('original');
            } else {
                switchModalView('repaired');
            }
        } else if (isResultsVisible) {
            // 结果页面的视图切换：修复后 -> 叶脉图 -> 修复前 -> 修复后
            let currentTab = 'repaired';
            if (!edgeGrid.classList.contains('hidden')) {
                currentTab = 'edge';
            } else if (!originalGrid.classList.contains('hidden')) {
                currentTab = 'original';
            }
            
            const tabOrder = ['repaired', 'edge', 'original'];
            const currentIndex = tabOrder.indexOf(currentTab);
            const nextIndex = (currentIndex + 1) % tabOrder.length;
            switchTab(tabOrder[nextIndex]);
        }
    }
});

function downloadSingle(index) {
    const result = batchResults[index];
    if (!result || !result.success) return;

    const a = document.createElement('a');
    a.href = result.repaired_image;
    a.download = result.filename.replace(/\.[^/.]+$/, '') + '_repaired.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function downloadEdgeImage(index) {
    const result = batchResults[index];
    if (!result || !result.success) return;

    const a = document.createElement('a');
    a.href = result.edge_image;
    a.download = result.filename.replace(/\.[^/.]+$/, '') + '_edge.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// 重新上传功能 - 直接刷新页面
function reuploadImages() {
    if (confirm('确定要重新上传新图片吗？当前所有数据和结果将被清除。')) {
        // 直接刷新页面，回到初始状态
        window.location.reload();
    }
}

async function downloadBatchResults() {
    if (batchResults.length === 0) {
        alert('没有可下载的结果！');
        return;
    }

    try {
        const response = await fetch('/api/download_batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ results: batchResults })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'repaired_images.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('下载失败！');
        }
    } catch (error) {
        console.error(error);
        alert('下载出错！');
    }
}

// Modal functions
function openModal(index) {
    currentModalIndex = index;
    currentModalView = 'repaired'; // Default to repaired view
    updateModalContent();
    imageModal.classList.add('show');
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
}

function closeModal() {
    imageModal.classList.remove('show');
    document.body.style.overflow = 'auto'; // Restore scrolling
}

function updateModalContent() {
    if (currentModalIndex < 0 || currentModalIndex >= batchResults.length) return;
    
    const result = batchResults[currentModalIndex];
    if (!result || !result.success) return;
    
    let imageUrl, titleText;
    
    switch(currentModalView) {
        case 'repaired':
            imageUrl = result.repaired_image;
            titleText = `修复后 - ${result.filename}`;
            break;
        case 'edge':
            imageUrl = result.edge_image;
            titleText = `SAIN 结构感知叶脉图 - ${result.filename}`;
            break;
        case 'original':
            imageUrl = uploadedImages[currentModalIndex].src;
            titleText = `修复前 - ${uploadedImages[currentModalIndex].file.name}`;
            break;
    }
    
    modalImage.src = imageUrl;
    modalTitle.textContent = titleText;
    
    // Update button states
    modalTabRepaired.classList.toggle('active', currentModalView === 'repaired');
    modalTabEdge.classList.toggle('active', currentModalView === 'edge');
    modalTabOriginal.classList.toggle('active', currentModalView === 'original');
}

function switchModalView(view) {
    currentModalView = view;
    updateModalContent();
}

// Close modal when clicking outside the image
imageModal.addEventListener('click', function(e) {
    if (e.target === imageModal) {
        closeModal();
    }
});
