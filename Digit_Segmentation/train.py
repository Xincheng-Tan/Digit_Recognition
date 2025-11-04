import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import HDF5Dataset, MultiTaskLeNet5, Logger, DiceLoss, ComboLoss, POOL_TYPE, DECODER_TYPE

HDF5_DIR = './synthetic_dataset'
TRAIN_FILE = os.path.join(HDF5_DIR, 'train_data.h5')
TEST_FILE = os.path.join(HDF5_DIR, 'test_data.h5')

MASK_KEY = 'masks_task2'

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_NAME = f'{POOL_TYPE}_{DECODER_TYPE}.pth'
TRAIN_DIR = "./train/"
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
WEIGHTS_DIR = "./weights/"
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


def train_model(model, train_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    # 分割损失：用于前景/背景 (2 类)
    seg_loss_fn = ComboLoss(num_classes=2, dice_weight=0.5, ce_weight=0.5).to(device)
    # 分类损失：用于数字 0-9 (10 类)
    class_loss_fn = nn.CrossEntropyLoss().to(device)
    LAMBDA_SEG = 1.0
    LAMBDA_CLASS = 1.0
    
    for i, (images, masks, target_class) in enumerate(train_loader):
        # images: (N, 1, 64, 64), masks: (N, 64, 64), target_class: (N, )
        images = images.to(device)
        masks = masks.to(device)
        target_class = target_class.to(device)

        seg_target = (masks > 0).long()
        
        # seg_pred: (N, 2, 64, 64), class_pred: (N, 10)
        seg_pred, class_pred = model(images)
        
        seg_loss = seg_loss_fn(seg_pred, seg_target)
        class_loss = class_loss_fn(class_pred, target_class)
        total_loss = LAMBDA_SEG * seg_loss + LAMBDA_CLASS * class_loss
        
        # 3. 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"   Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                  f"Total Loss: {total_loss.item():.4f} (Seg: {seg_loss.item():.4f}, Class: {class_loss.item():.4f})")

    avg_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1} 结束, 平均训练 Total Loss: {avg_loss:.4f} ---")


sys.stdout = Logger(f'{POOL_TYPE}_{DECODER_TYPE}_train.log', sys.stdout, DIR=TRAIN_DIR)
try:
    print(f"--- Multi-Task FCN-LeNet5 训练 ---")
    print(f"    DECODER_TYPE = {DECODER_TYPE}")
    print(f"    DEVICE = {DEVICE}")
    
    print("\n1. 正在加载数据集...")
    train_dataset = HDF5Dataset(TRAIN_FILE, mask_key=MASK_KEY)
    test_dataset = HDF5Dataset(TEST_FILE, mask_key=MASK_KEY)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4 if DEVICE.type == 'cuda' else 0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4 if DEVICE.type == 'cuda' else 0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print("\n2. 正在初始化模型...")
    model = MultiTaskLeNet5(
        n_seg_classes=2, 
        n_class_classes=10,
        decoder_type=DECODER_TYPE
    ).to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n3. 正在开始训练...")
    for epoch in range(NUM_EPOCHS):
        train_model(model, train_loader, optimizer, DEVICE, epoch)
    print("✅ 训练完成。")
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, MODEL_SAVE_NAME))
    print(f"\n✅ 模型保存在: {WEIGHTS_DIR}{MODEL_SAVE_NAME}")

except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    raise

finally:
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close()