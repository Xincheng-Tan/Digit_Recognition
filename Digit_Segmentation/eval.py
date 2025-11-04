import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import HDF5Dataset, MultiTaskLeNet5, Logger, DiceLoss, ComboLoss, POOL_TYPE, DECODER_TYPE

HDF5_DIR = './synthetic_dataset'
TRAIN_FILE = os.path.join(HDF5_DIR, 'train_data.h5')
TEST_FILE = os.path.join(HDF5_DIR, 'test_data.h5')

SEG_CLASSES = 2
CLASS_CLASSES = 10
MASK_KEY = 'masks_task2' 

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_NAME = f'{POOL_TYPE}_{DECODER_TYPE}.pth' 
TEST_DIR = "./test/"
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
WEIGHTS_DIR = "./weights/"
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


def calculate_iou(seg_pred, seg_target, n_classes, smooth=1e-6):
    """
    计算 Intersection over Union (IoU) / Jaccard Index
    Args:
        seg_pred (torch.Tensor): 分割预测结果, (N, H, W), 值为类别索引 (0到N_CLASSES-1)
        seg_target (torch.Tensor): 真实分割掩码, (N, H, W), 值为类别索引 (0到N_CLASSES-1)
        n_classes (int): 类别数量 (应为 2)
        smooth (float): 平滑项
    Returns:
        tuple: (IoU_per_class, mIoU)
    """

    seg_pred = seg_pred.view(-1)
    seg_target = seg_target.view(-1)
    
    iou_list = []
    for cls in range(n_classes):
        pred_mask = (seg_pred == cls)
        target_mask = (seg_target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)

    iou_per_class = np.array(iou_list)
    m_iou = np.mean(iou_per_class)

    return iou_per_class, m_iou


def calculate_dice(seg_pred, seg_target, n_classes, smooth=1e-6):
    """
    计算 Dice 系数 (Sørensen–Dice coefficient)
    Args:
        seg_pred (torch.Tensor): 分割预测结果, (N, H, W), 值为类别索引 (0到N_CLASSES-1)
        seg_target (torch.Tensor): 真实分割掩码, (N, H, W), 值为类别索引 (0到N_CLASSES-1)
        n_classes (int): 类别数量 (应为 2)
        smooth (float): 平滑项
    Returns:
        tuple: (Dice_per_class, mDice)
    """
    seg_pred = seg_pred.view(-1)
    seg_target = seg_target.view(-1)
    
    dice_list = []
    for cls in range(n_classes):
        pred_mask = (seg_pred == cls)
        target_mask = (seg_target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        A = pred_mask.sum().item()
        B = target_mask.sum().item()
        
        # Dice = (2 * Intersection) / (A + B)
        dice = (2. * intersection + smooth) / (A + B + smooth)
        dice_list.append(dice)

    dice_per_class = np.array(dice_list)
    m_dice = np.mean(dice_per_class)

    return dice_per_class, m_dice


def test_model(model, test_loader, device):
    model.eval()
    
    seg_loss_fn = ComboLoss(num_classes=SEG_CLASSES).to(device)
    class_loss_fn = nn.CrossEntropyLoss().to(device)
    LAMBDA_SEG = 1.0
    LAMBDA_CLASS = 1.0
    
    test_loss_total = 0.0
    test_loss_seg = 0.0
    test_loss_class = 0.0
    
    total_seg_iou_per_class = np.zeros(SEG_CLASSES)
    total_seg_dice_per_class = np.zeros(SEG_CLASSES)
    correct_class_predictions = 0
    total_class_samples = 0
    
    total_batches = 0
    
    with torch.no_grad():
        for images, masks_task2, target_class in test_loader:
            images = images.to(device)
            masks_task2 = masks_task2.to(device)
            target_class = target_class.to(device)
            
            seg_target = (masks_task2 > 0).long()
            
            seg_pred_logits, class_pred_logits = model(images)
            
            seg_loss = seg_loss_fn(seg_pred_logits, seg_target)
            class_loss = class_loss_fn(class_pred_logits, target_class)
            total_loss = LAMBDA_SEG * seg_loss + LAMBDA_CLASS * class_loss
            
            test_loss_total += total_loss.item()
            test_loss_seg += seg_loss.item()
            test_loss_class += class_loss.item()
            
            _, seg_predicted = torch.max(seg_pred_logits.data, 1) # (N, H, W)
            
            iou_per_class_batch, _ = calculate_iou(seg_predicted, seg_target, SEG_CLASSES)
            total_seg_iou_per_class += iou_per_class_batch
            
            dice_per_class_batch, _ = calculate_dice(seg_predicted, seg_target, SEG_CLASSES)
            total_seg_dice_per_class += dice_per_class_batch
            
            _, class_predicted = torch.max(class_pred_logits.data, 1) # (N)
            correct_class_predictions += (class_predicted == target_class).sum().item()
            total_class_samples += target_class.size(0)
            
            total_batches += 1

    avg_loss_total = test_loss_total / len(test_loader)
    avg_loss_seg = test_loss_seg / len(test_loader)
    avg_loss_class = test_loss_class / len(test_loader)
    
    mean_iou_per_class = total_seg_iou_per_class / total_batches
    m_iou = np.mean(mean_iou_per_class)
    
    mean_dice_per_class = total_seg_dice_per_class / total_batches
    m_dice = np.mean(mean_dice_per_class)
    
    class_accuracy = 100 * correct_class_predictions / total_class_samples

    print(f"\n✅ 测试结果:")
    print(f"     平均 Total Loss: {avg_loss_total:.4f} (Seg: {avg_loss_seg:.4f}, Class: {avg_loss_class:.4f})")
    
    print("\n--- 分割任务 (前景/背景) ---")
    print(f"     Mean IoU (mIoU): {m_iou:.4f}")
    print(f"     - Class 0 (Background) IoU: {mean_iou_per_class[0]:.4f}")
    print(f"     - Class 1 (Foreground) IoU: {mean_iou_per_class[1]:.4f}")
    
    print(f"\n     Mean Dice Coefficient (mDice): {m_dice:.4f}")
    print(f"     - Class 0 (Background) Dice: {mean_dice_per_class[0]:.4f}")
    print(f"     - Class 1 (Foreground) Dice: {mean_dice_per_class[1]:.4f}")
    
    print("\n--- 分类任务 (数字 0-9) ---")
    print(f"     Image Classification Accuracy: {class_accuracy:.2f}%")


def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    
    test_iter = iter(test_loader)
    BATCH_INDEX_TO_VIEW = 4
    for _ in range(BATCH_INDEX_TO_VIEW):
        next(test_iter)

    images, masks_task2, gt_class = next(test_iter)
    
    images = images.to(device)
    masks_task2 = masks_task2.to(device)
    
    with torch.no_grad():
        seg_pred_logits, class_pred_logits = model(images)
    
    _, seg_preds = torch.max(seg_pred_logits, 1)        # 2类分割预测 (0/1)
    _, class_preds = torch.max(class_pred_logits, 1)    # 10类数字分类预测 (0-9)

    images_cpu = images.cpu().numpy()
    gt_11_class = masks_task2.cpu().numpy() 
    seg_preds_bool = (seg_preds.cpu() == 1) # (N, H, W) Bool
    
    predicted_digit_value = (class_preds.cpu() + 1).unsqueeze(1).unsqueeze(2) # (N, 1, 1) -> (N, H, W)
    pred_11_class = (seg_preds_bool * predicted_digit_value).numpy()
    
    vmin = 0
    vmax = CLASS_CLASSES # 10 (对应 0-10 类别)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5)) 
    fig.suptitle(f"Multi-Task Segmentation & Classification ({POOL_TYPE} | Decoder: {DECODER_TYPE})", fontsize=16)

    for i in range(num_samples):
        ax = axes[i, 0]
        ax.imshow(images_cpu[i, 0], cmap='gray')
        ax.set_title(f"Input #{i}")
        ax.axis('off')
        
        # 2. Segmentation GT
        ax = axes[i, 1]
        ax.imshow(gt_11_class[i], cmap='tab20', vmin=vmin, vmax=vmax)
        ax.set_title(f"Seg GT | Class: {gt_class[i].item()}") 
        ax.axis('off')
        
        # 3. Segmentation Predicted 
        ax = axes[i, 2]
        ax.imshow(pred_11_class[i], cmap='tab20', vmin=vmin, vmax=vmax)
        ax.set_title(f"Seg Pred | Class: {class_preds[i].item()}")
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(TEST_DIR, f"{POOL_TYPE}_{DECODER_TYPE}.png")) 
    print(f"✅ 可视化结果保存在 ./test/{POOL_TYPE}_{DECODER_TYPE}.png")


sys.stdout = Logger(f'{POOL_TYPE}_{DECODER_TYPE}_test.log', sys.stdout, DIR=TEST_DIR)
try:
    print(f"--- Multi-Task FCN-LeNet5 测试 ---")
    print(f"    分割类别: {SEG_CLASSES}, 分类类别: {CLASS_CLASSES}")
    print(f"    DECODER_TYPE = {DECODER_TYPE}, DEVICE = {DEVICE}")

    print("\n1. 正在加载测试数据集...")
    test_dataset = HDF5Dataset(TEST_FILE, mask_key=MASK_KEY) # 加载 masks_task2 (0-10)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4 if DEVICE.type == 'cuda' else 0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print("\n2. 正在初始化模型并加载权重...")
    model = MultiTaskLeNet5(
        n_seg_classes=SEG_CLASSES, 
        n_class_classes=CLASS_CLASSES, 
        decoder_type=DECODER_TYPE
    ).to(DEVICE)

    model_path = os.path.join(WEIGHTS_DIR, MODEL_SAVE_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}。请先运行训练阶段")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"✅ 模型权重已从 {model_path} 成功加载")

    print("\n3. 正在测试模型...")
    test_model(model, test_loader, DEVICE)

    print("\n4. 正在生成可视化结果...")
    visualize_predictions(model, test_loader, DEVICE, num_samples=5)

except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    raise

finally:
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close()