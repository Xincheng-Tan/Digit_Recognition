import os
import random
import h5py
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import cv2

HDF5_DIR = './synthetic_dataset'
TRAIN_FILE = os.path.join(HDF5_DIR, 'train_data.h5')
TEST_FILE = os.path.join(HDF5_DIR, 'test_data.h5')
TOTAL_SAMPLES = 200000 # 设定的总样本数
TEST_SPLIT_RATIO = 0.1 # 测试集比例 10%

PATCH_SIZE = 64
DIGIT_SIZE = 28
BACKGROUND_DIR = './data/pic'
MNIST_DATA_PATH = './data'


def load_and_prepare_mnist(root_path: str = MNIST_DATA_PATH):
    """加载并准备 MNIST 数据，提取前景图和掩码"""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=root_path, train=True, download=True, transform=transform)
    prepared_data = []
    for image_tensor, label in mnist_dataset:
        mnist_image = image_tensor.squeeze(0).numpy()
        digit_fg = mnist_image.copy()
        digit_mask = (mnist_image > 0).astype(np.float32)
        prepared_data.append({'fg': digit_fg, 'mask': digit_mask, 'label': label})
    return prepared_data


def load_background_images(bg_dir: str = BACKGROUND_DIR) -> List[np.ndarray]:
    """加载背景图片并转换为灰度图 NumPy 数组"""
    bg_images = []
    for filename in os.listdir(bg_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(bg_dir, filename)).convert('L')
            img_array = np.array(img, dtype=np.uint8)
            bg_images.append(img_array)
    return bg_images


def apply_augmentation(digit_fg: np.ndarray, digit_mask: np.ndarray, 
                       max_angle: float = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    对前景图和掩码应用随机旋转。
    返回:
        augmented_fg: 增强后的前景图 (28x28)
        augmented_mask: 增强后的掩码 (28x28)
    """
    H, W = DIGIT_SIZE, DIGIT_SIZE
    center = (W/2 - 0.5, H/2 - 0.5)
    angle = random.uniform(-max_angle, max_angle)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented_fg = cv2.warpAffine(digit_fg, M_rot, (W, H), 
                                  flags=cv2.INTER_LINEAR, borderValue=0.0)
    augmented_mask = cv2.warpAffine(digit_mask, M_rot, (W, H), 
                                    flags=cv2.INTER_NEAREST, borderValue=0.0)
    augmented_mask = (augmented_mask > 0).astype(np.float32)   
    return augmented_fg, augmented_mask


def create_synthetic_sample(prepared_mnist_data: List[Dict], bg_images: List[np.ndarray], 
                            patch_size: int = PATCH_SIZE) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    从背景图和 MNIST 数字中创建单个合成样本，并应用数据增强。
    返回: synthetic_image, gt_mask_task1, gt_mask_task2, label
    """
    H, W = patch_size, patch_size
    
    if not bg_images: return None
    
    # 1. 裁剪背景图
    bg_image = random.choice(bg_images)
    bg_H, bg_W = bg_image.shape
    if bg_H < H or bg_W < W: return None

    start_y = random.randint(0, bg_H - H)
    start_x = random.randint(0, bg_W - W)
    bg_patch = bg_image[start_y:start_y + H, start_x:start_x + W]
    bg_patch_float = bg_patch.astype(np.float32) / 255.0

    # 2. 选择 MNIST 样本并应用数据增强
    mnist_sample = random.choice(prepared_mnist_data)
    digit_fg_orig = mnist_sample['fg']
    digit_mask_orig = mnist_sample['mask']
    label = mnist_sample['label']
    digit_fg, digit_mask = apply_augmentation(digit_fg_orig, digit_mask_orig)
    
    # 3. 随机定位数字在 Patch 中的位置
    max_x = W - DIGIT_SIZE
    max_y = H - DIGIT_SIZE
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)
    
    # 4. 叠加数字到背景 Patch
    synthetic_image = bg_patch_float.copy()
    roi = synthetic_image[pos_y:pos_y + DIGIT_SIZE, pos_x:pos_x + DIGIT_SIZE]
    
    new_roi = roi * (1.0 - digit_mask) + digit_fg * digit_mask
    synthetic_image[pos_y:pos_y + DIGIT_SIZE, pos_x:pos_x + DIGIT_SIZE] = new_roi
    
    gt_mask_task1 = np.zeros((H, W), dtype=np.uint8) # 用于任务1 (前景/背景)
    gt_mask_task2 = np.zeros((H, W), dtype=np.uint8) # 用于任务2 (多类别)
    
    # 任务1：前景背景分割 (2分类)
    gt_mask_task1[pos_y:pos_y + DIGIT_SIZE, pos_x:pos_x + DIGIT_SIZE] = (digit_mask > 0).astype(np.uint8)
    
    # 任务2：按照数字类别分割 (11分类: 0-背景, 1-数字0, 2-数字1, ..., 10-数字9)
    mask_value = label + 1 # 类别映射：0->1, 1->2, ..., 9->10
    
    gt_mask_task2[pos_y:pos_y + DIGIT_SIZE, pos_x:pos_x + DIGIT_SIZE] = (digit_mask * mask_value).astype(np.uint8)

    return synthetic_image, gt_mask_task1, gt_mask_task2, label


def generate_and_save_dataset(total_samples: int, test_split_ratio: float):
    """生成数据集并保存到 HDF5 文件。"""
    os.makedirs(HDF5_DIR, exist_ok=True)
    
    print("⏳ 1. 正在加载 MNIST 数据...")
    prepared_mnist_data = load_and_prepare_mnist()
    print("⏳ 2. 正在加载背景图片...")
    bg_images = load_background_images()
    
    if not prepared_mnist_data or not bg_images:
        print("🛑 缺少必要的数据源 (MNIST 或背景图)。请检查路径。")
        return

    print(f"⏳ 3. 正在生成 {total_samples} 个样本 (含数据增强)...")
    
    all_images = []
    all_masks_task1 = []
    all_masks_task2 = []
    all_labels = [] # <--- 新增：收集标签
    
    count = 0
    while count < total_samples:
        # 修改：接收 label
        sample = create_synthetic_sample(prepared_mnist_data, bg_images, PATCH_SIZE)
        if sample:
            img, mask1, mask2, label = sample # <--- 解包 label
            all_images.append(img)
            all_masks_task1.append(mask1)
            all_masks_task2.append(mask2)
            all_labels.append(label) # <--- 存储 label
            count += 1
            if count % 1000 == 0 or count == total_samples:
                print(f"   已生成 {count}/{total_samples} 个样本...")
        else:
            continue

    print("⏳ 4. 正在转换数据并划分训练/测试集...")
    
    all_images = np.array(all_images, dtype=np.float32)
    all_masks_task1 = np.array(all_masks_task1, dtype=np.uint8)
    all_masks_task2 = np.array(all_masks_task2, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.uint8) # <--- 转换 labels
    
    num_test = int(total_samples * test_split_ratio)
    num_train = total_samples - num_test
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_images = all_images[train_indices]
    train_masks_task1 = all_masks_task1[train_indices]
    train_masks_task2 = all_masks_task2[train_indices]
    train_labels = all_labels[train_indices] # <--- 划分 labels
    
    test_images = all_images[test_indices]
    test_masks_task1 = all_masks_task1[test_indices]
    test_masks_task2 = all_masks_task2[test_indices]
    test_labels = all_labels[test_indices] # <--- 划分 labels
    
    print(f"   训练集数量: {num_train}, 测试集数量: {num_test}")
    print("⏳ 5. 正在保存 HDF5 文件...")

    def save_to_hdf5(file_path, images, masks_task1, masks_task2, labels):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('images', data=images, compression="gzip", compression_opts=9)
            f.create_dataset('masks_task1', data=masks_task1, compression="gzip", compression_opts=9)
            f.create_dataset('masks_task2', data=masks_task2, compression="gzip", compression_opts=9)
            f.create_dataset('labels', data=labels, compression="gzip", compression_opts=9) # <--- 新增：保存 labels
        print(f"✅ 保存到: {file_path}")

    save_to_hdf5(TRAIN_FILE, train_images, train_masks_task1, train_masks_task2, train_labels) # <--- 传入 labels
    save_to_hdf5(TEST_FILE, test_images, test_masks_task1, test_masks_task2, test_labels) # <--- 传入 labels
    
    print("🎉 数据集生成完毕！")

if __name__ == "__main__":
    generate_and_save_dataset(TOTAL_SAMPLES, TEST_SPLIT_RATIO)