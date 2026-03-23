import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def generate_irregular_mask(size=256, max_lines=10):
    mask = np.zeros((size, size), np.uint8)
    for _ in range(random.randint(1, max_lines)):
        # 随机画一些粗细不一的线，模拟涂抹
        x1, y1 = random.randint(0, size), random.randint(0, size)
        x2, y2 = random.randint(0, size), random.randint(0, size)
        thickness = random.randint(5, 20)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        # 随机画几个圈
        cv2.circle(mask, (random.randint(0, size), random.randint(0, size)),
                   random.randint(5, 30), 255, -1)
    return mask

def main():
    save_path = '/root/autodl-tmp/data/celeba_mask'
    os.makedirs(save_path, exist_ok=True)
    print("正在生成 10,000 张随机掩码...")
    for i in tqdm(range(10000)):
        mask = generate_irregular_mask()
        cv2.imwrite(os.path.join(save_path, f'mask_{i:05d}.png'), mask)

if __name__ == "__main__":
    main()