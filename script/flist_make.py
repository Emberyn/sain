import os


def generate_flist(image_folder, output_file):
    abs_image_folder = os.path.abspath(image_folder)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(abs_image_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_list.append(os.path.join(root, file))

    file_list.sort()  # 排序确保序号对齐
    with open(output_file, 'w') as f:
        for path in file_list:
            f.write(path + '\n')
    print(f"成功生成: {os.path.basename(output_file)} (共 {len(file_list)} 张图)")


if __name__ == "__main__":
    # 基础路径
    base_data = "/data/chen/sain/dataset/DataSet_padded_enhance"
    base_out = "/data/chen/sain/checkpoint"

    # 任务清单：(源文件夹, 输出文件名)
    tasks = [
        # --- 原图与边缘 ---
        (f"{base_data}/train_padded_enhance", f"{base_out}/train_images.flist"),
        (f"{base_data}/test_padded_enhance", f"{base_out}/test_images.flist"),
        (f"{base_data}/3ch/train_padded_enhance_edge_3ch", f"{base_out}/train_edges.flist"),
        (f"{base_data}/3ch/test_padded_enhance_edge_3ch", f"{base_out}/test_edges.flist"),

        # --- 掩码：训练集 (Train Masks) ---
        (f"{base_data}/mask/mask_train", f"{base_out}/train_masks_mixed.flist"),  # 混合难度
        (f"{base_data}/mask/mask_train_10_20", f"{base_out}/train_masks_easy.flist"),  # 简单/中等
        (f"{base_data}/mask/mask_train_40_60", f"{base_out}/train_masks_hard.flist"),  # 困难

        # --- 掩码：测试集 (Test Masks) ---
        (f"{base_data}/mask/mask_test", f"{base_out}/test_masks_mixed.flist"),
        (f"{base_data}/mask/mask_test_10_20", f"{base_out}/test_masks_easy.flist"),
        (f"{base_data}/mask/mask_test_40_60", f"{base_out}/test_masks_hard.flist"),
    ]

    for src, out in tasks:
        if os.path.exists(src):
            generate_flist(src, out)
        else:
            print(f"跳过：找不到目录 {src}")