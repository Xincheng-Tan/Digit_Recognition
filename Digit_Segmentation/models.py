import os
import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# 'interpolate' (FCN-32s 风格) 或 'transpose' (可学习的反卷积)
DECODER_TYPE = 'interpolate'  
# DECODER_TYPE = 'transpose'
POOL_TYPE = 'maxpool'  
# POOL_TYPE = 'avgpool'


class Logger(object):
    def __init__(self, filename="Default.log", stream=sys.stdout, DIR="./"):
        self.terminal = stream
        self.log = open(os.path.join(DIR, f'{filename}'), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        self.log.flush() 

    def flush(self):
        pass
        # self.terminal.flush()


class HDF5Dataset(Dataset):
    def __init__(self, file_path, mask_key='masks_task2'):
        self.file_path = file_path
        self.mask_key = mask_key
        
        with h5py.File(self.file_path, 'r') as f:
            self.images = f['images'][:]
            self.masks = f[self.mask_key][:]
            self.labels = f['labels'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] # (64, 64)
        mask = self.masks[idx]   # (64, 64)
        labels = self.labels[idx]
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        label = torch.tensor(labels, dtype=torch.long)
        
        return image_tensor, mask_tensor, label


class MultiTaskLeNet5(nn.Module):
    def __init__(self, n_seg_classes=2, n_class_classes=10, decoder_type='transpose'):
        super(MultiTaskLeNet5, self).__init__()
        
        self.n_seg_classes = n_seg_classes
        self.n_class_classes = n_class_classes
        self.decoder_type = decoder_type
        
        # --- 1. 编码器 (共享 LeNet-5 结构) ---
        # 64x64 -> 32x32 (S2) -> 16x16 (S4)
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        if POOL_TYPE == 'maxpool':
            self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        if POOL_TYPE == 'maxpool':
            self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # --- 2. 两个 Heads ---
        # A. 分割头 (Segmentation Head) - FCN 结构
        # 输入: (N, 16, 16, 16)
        self.seg_c5 = nn.Conv2d(16, 120, kernel_size=1) 
        self.seg_relu3 = nn.ReLU()
        self.seg_f6 = nn.Conv2d(120, 84, kernel_size=1)
        self.seg_relu4 = nn.ReLU()
        # 输出: (N, n_seg_classes, 16, 16)
        self.seg_score = nn.Conv2d(84, self.n_seg_classes, kernel_size=1)

        # B. 分类头 (Classification Head) - FC 结构
        # 沿用编码器 S4 的特征作为输入，但会进行全局池化
        self.class_conv = nn.Conv2d(16, 32, kernel_size=1) # 16x16x16 -> 16x16x32
        self.class_relu = nn.ReLU()
        # (N, 32, 16, 16) -> (N, 32, 1, 1) -> (N, 32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # (N, 32) -> (N, n_class_classes)
        self.class_fc = nn.Linear(32, self.n_class_classes)

        # --- 3. 分割解码器 (仅针对 Seg Head) ---
        if self.decoder_type == 'transpose':
            # S4 (16x16) -> 2 classes
            self.seg_score_s4 = nn.Conv2d(16, self.n_seg_classes, kernel_size=1) 
            # 16x16 -> 32x32
            self.seg_upscore_l1 = nn.ConvTranspose2d(
                self.n_seg_classes, self.n_seg_classes, kernel_size=4, stride=2, padding=1, bias=False)
            # S2 (32x32) -> 2 classes
            self.seg_score_s2 = nn.Conv2d(6, self.n_seg_classes, kernel_size=1) 
            # 32x32 -> 64x64
            self.seg_upscore_l2 = nn.ConvTranspose2d(
                self.n_seg_classes, self.n_seg_classes, kernel_size=4, stride=2, padding=1, bias=False)


    def forward(self, x):
        # 64x64
        x = self.relu1(self.c1(x))
        s2_features = self.s2(x) # 32x32, 6c
        s2_features = self.dropout1(s2_features)

        # 32x32
        x = self.relu2(self.c3(s2_features))
        s4_features = self.s4(x) # 16x16, 16c
        s4_features = self.dropout2(s4_features)

        # --- 2. 分类任务 (Classification Head) ---
        class_x = self.class_relu(self.class_conv(s4_features))
        # (N, 32, 16, 16) -> (N, 32, 1, 1)
        class_x = self.avgpool(class_x) 
        # (N, 32) -> (N, 10)
        class_out = self.class_fc(class_x.view(class_x.size(0), -1))

        # --- 3. 分割任务 (Segmentation Head) ---
        # FCN Head: 16x16, 16c -> 16x16, n_seg_classes
        seg_x = self.seg_relu3(self.seg_c5(s4_features))
        seg_x = self.seg_relu4(self.seg_f6(seg_x))
        seg_score_pool4 = self.seg_score(seg_x) # 16x16, n_seg_classes
        
        if self.decoder_type == 'transpose':
            # (1) FCN-16s 跳跃连接：融合 S4 层
            seg_score_s4 = self.seg_score_s4(s4_features) # 16x16, 2c
            seg_upscore_1 = self.seg_upscore_l1(seg_score_pool4) # 32x32   
            seg_score_s4_upsampled = F.interpolate(
                seg_score_s4, size=seg_upscore_1.shape[2:], mode='bilinear', align_corners=False)
            fused_score_l1 = seg_upscore_1 + seg_score_s4_upsampled
            
            # (2) FCN-8s 跳跃连接：融合 S2 层
            seg_score_s2 = self.seg_score_s2(s2_features) # 32x32, 2c
            seg_final_upscore = self.seg_upscore_l2(fused_score_l1) # 64x64
            seg_score_s2_upsampled = F.interpolate(
                seg_score_s2, size=seg_final_upscore.shape[2:], mode='bilinear', align_corners=False)
            seg_out = seg_final_upscore + seg_score_s2_upsampled

        elif self.decoder_type == 'interpolate':
            # 直接 4x 双线性插值
            seg_out = F.interpolate(
                seg_score_pool4, scale_factor=4, mode='bilinear', align_corners=False)

        return seg_out, class_out


class ResidualBlock(nn.Module):
    """
    一个标准的残差块，支持步幅、空洞和下采样。
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # K=3, D=1, P=1  => 保持尺寸
        # K=3, D=2, P=2  => 保持尺寸
        padding = dilation   
        # 卷积层 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 卷积层 2 (stride 始终为 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                             stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样层 (用于 shortcut connection)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        
        return out


class DilatedResidualLeNet(nn.Module):
    def __init__(self, n_seg_classes=2, n_class_classes=10, decoder_type='transpose'):
        super(DilatedResidualLeNet, self).__init__()
        
        self.n_seg_classes = n_seg_classes
        self.n_class_classes = n_class_classes
        self.decoder_type = decoder_type
        
        # --- 1. 编码器 (Dilated Residual Encoder) ---
        # 64x64 -> 32x32 (s2) -> 16x16 (s4)
        # 1x64x64 -> 6x64x64
        self.c1 = nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=True)
        # 6x64x64 -> 6x32x32
        self.res_block1 = self._make_residual_layer(6, 6, stride=2)
        # 6x32x32 -> 16x16x16
        self.res_block2 = self._make_residual_layer(6, 16, stride=2)
        # 16x16x16 -> 16x16x16 (感受野增大，但分辨率不变)
        self.res_block3_dilated = self._make_residual_layer(16, 16, stride=1, dilation=2)
        
        # --- 2. 两个 Heads ---
        # A. 分割头 (Segmentation Head)
        # 输入: (N, 16, 16, 16)
        self.seg_c5 = nn.Conv2d(16, 120, kernel_size=1) 
        self.seg_relu3 = nn.ReLU()
        self.seg_f6 = nn.Conv2d(120, 84, kernel_size=1)
        self.seg_relu4 = nn.ReLU()
        self.seg_score = nn.Conv2d(84, self.n_seg_classes, kernel_size=1)
        # B. 分类头 (Classification Head)
        self.class_conv = nn.Conv2d(16, 32, kernel_size=1) 
        self.class_relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_fc = nn.Linear(32, self.n_class_classes)

        # --- 3. 分割解码器 ---
        if self.decoder_type == 'transpose':
            # S4 (16x16, 16c) -> 2 classes
            self.seg_score_s4 = nn.Conv2d(16, self.n_seg_classes, kernel_size=1) 
            self.seg_upscore_l1 = nn.ConvTranspose2d(
                self.n_seg_classes, self.n_seg_classes, kernel_size=4, stride=2, padding=1, bias=False)
            # S2 (32x32, 6c) -> 2 classes
            self.seg_score_s2 = nn.Conv2d(6, self.n_seg_classes, kernel_size=1) 
            self.seg_upscore_l2 = nn.ConvTranspose2d(
                self.n_seg_classes, self.n_seg_classes, kernel_size=4, stride=2, padding=1, bias=False)

    def _make_residual_layer(self, in_channels, out_channels, stride, dilation=1):
        """
        构建一个残差块，自动处理 shortcut connection 的下采样。
        """
        downsample = None
        # 当步幅不为1，或输入输出通道数不同时，需要添加下采样层 (1x1 卷积)
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        return ResidualBlock(in_channels, out_channels, stride, dilation, downsample)


    def forward(self, x):
        # --- 1. 编码器 ---
        # 64x64
        x = self.relu(self.bn1(self.c1(x))) # (N, 6, 64, 64)
        # 64x64 -> 32x32
        s2_features = self.res_block1(x) # (N, 6, 32, 32)
        # 32x32 -> 16x16
        s4_base_features = self.res_block2(s2_features) # (N, 16, 16, 16)
        # 16x16 -> 16x16 (空洞卷积)
        s4_features = self.res_block3_dilated(s4_base_features) # (N, 16, 16, 16)

        # --- 2. 分类任务 (Classification Head) ---
        class_x = self.class_relu(self.class_conv(s4_features))
        class_x = self.avgpool(class_x) 
        class_out = self.class_fc(class_x.view(class_x.size(0), -1))

        # --- 3. 分割任务 (Segmentation Head) ---
        seg_x = self.seg_relu3(self.seg_c5(s4_features))
        seg_x = self.seg_relu4(self.seg_f6(seg_x))
        seg_score_pool4 = self.seg_score(seg_x) 
        
        if self.decoder_type == 'transpose':
            # (1) FCN-16s 跳跃连接：融合 S4 层
            seg_score_s4 = self.seg_score_s4(s4_features) 
            seg_upscore_1 = self.seg_upscore_l1(seg_score_pool4) 
            seg_score_s4_upsampled = F.interpolate(
                seg_score_s4, size=seg_upscore_1.shape[2:], mode='bilinear', align_corners=False)
            fused_score_l1 = seg_upscore_1 + seg_score_s4_upsampled
            
            # (2) FCN-8s 跳跃连接：融合 S2 层
            seg_score_s2 = self.seg_score_s2(s2_features)
            seg_final_upscore = self.seg_upscore_l2(fused_score_l1)
            seg_score_s2_upsampled = F.interpolate(
                seg_score_s2, size=seg_final_upscore.shape[2:], mode='bilinear', align_corners=False)
            seg_out = seg_final_upscore + seg_score_s2_upsampled

        elif self.decoder_type == 'interpolate':
            seg_out = F.interpolate(
                seg_score_pool4, scale_factor=4, mode='bilinear', align_corners=False)

        return seg_out, class_out
    

class DiceLoss(nn.Module):
    """
    多类别 Soft Dice Loss 的实现
    """
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth    # 平滑项，防止除以零

    def forward(self, pred, target):
        """
        计算多类别 Dice Loss
        Args:
            pred (torch.Tensor): 模型的预测输出 (Logits)，形状为 (N, C, H, W)
            target (torch.Tensor): 真实标签，形状为 (N, H, W)，包含类别索引 (0 到 C-1)
        Returns:
            torch.Tensor: Dice Loss 的平均值
        """
        # 1. 将 Logits 转换为 Softmax 概率
        # (N, C, H, W) -> (N, C, H, W)
        pred_softmax = F.softmax(pred, dim=1)
        
        # 2. 将目标 (target) 转换为 One-Hot 编码
        # target (N, H, W) -> target_one_hot (N, C, H, W)
        # target 必须是 LongTensor
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        assert pred_softmax.shape == target_one_hot.shape
        
        dice_losses = []
        # 某些实现会跳过背景类，这里为了完整性，我们包含所有类别
        for c in range(self.num_classes):
            pred_c = pred_softmax[:, c, :, :]
            target_c = target_one_hot[:, c, :, :]

            intersection = torch.sum(pred_c * target_c)
            union = torch.sum(pred_c) + torch.sum(target_c)

            # Dice 系数: (2 * intersection + smooth) / (union + smooth)
            dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)
            
            # Dice Loss = 1 - Dice Coefficient
            dice_loss = 1.0 - dice_coefficient
            dice_losses.append(dice_loss)

        return torch.stack(dice_losses).mean()


class ComboLoss(nn.Module):
    """
    Dice Loss 和 Cross-Entropy Loss 的组合
    """
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5):
        super(ComboLoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce
    
