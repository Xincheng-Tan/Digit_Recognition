import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

plt.rcParams['axes.unicode_minus'] = False 

LOG_DIR = './train/'
LOG_FILENAMES = [
    'maxpool_interpolate_train.log',
    'maxpool_transpose_train.log',
    'avgpool_interpolate_train.log',
    'avgpool_transpose_train.log'
]
PLT_DIR = "./plt/"
if not os.path.exists(PLT_DIR):
    os.makedirs(PLT_DIR)

PATTERN = re.compile(
    r"Epoch \[(\d+)/20\], Step \[\d+/\d+\], .*? \(Seg: (\d+\.\d+), Class: (\d+\.\d+)\)"
)
all_model_data = {}


def extract_and_calculate_avg_losses(log_filepath, pattern):
    """
    读取日志文件，提取每一步的损失，并计算每个 epoch 的平均损失。
    返回: (even_epochs, avg_seg_losses, avg_class_losses)
    """
    epoch_seg_losses = defaultdict(list)
    epoch_class_losses = defaultdict(list)
    
    with open(log_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                seg_loss = float(match.group(2))
                class_loss = float(match.group(3))
                
                epoch_seg_losses[epoch].append(seg_loss)
                epoch_class_losses[epoch].append(class_loss)

    even_epochs = []
    avg_seg_losses = []
    avg_class_losses = []

    for epoch in range(1, 21):
        if epoch in epoch_seg_losses and len(epoch_seg_losses[epoch]) > 0:
            avg_seg = np.mean(epoch_seg_losses[epoch])
            avg_class = np.mean(epoch_class_losses[epoch])
            
            even_epochs.append(epoch)
            avg_seg_losses.append(avg_seg)
            avg_class_losses.append(avg_class)

    return even_epochs, avg_seg_losses, avg_class_losses


print("🚀 开始处理日志文件...")
for filename in LOG_FILENAMES:
    log_filepath = os.path.join(LOG_DIR, filename)
    model_name = filename.replace('_train.log', '') # 提取模型名称作为标签

    epochs, seg_losses, class_losses = extract_and_calculate_avg_losses(log_filepath, PATTERN)
    
    if epochs is not None:
        all_model_data[model_name] = {
            'epochs': epochs,
            'seg_losses': seg_losses,
            'class_losses': class_losses
        }

plt.figure(figsize=(12, 7))
plt.title('Mean Segmentation Loss Curves (Comparison)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Seg Loss', fontsize=12)

markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']
colors = ['b', 'r', 'g', 'm']

max_epochs = 20

for i, (model_name, data) in enumerate(all_model_data.items()):
    plt.plot(
        data['epochs'], 
        data['seg_losses'], 
        marker=markers[i % len(markers)], 
        linestyle=linestyles[i % len(linestyles)], 
        color=colors[i % len(colors)], 
        label=model_name
    )

plt.xticks(range(1, max_epochs + 1))
plt.legend(title='Model')
plt.grid(True)
seg_plt_filename = 'combined_seg_loss.png'
plt.savefig(os.path.join(PLT_DIR, seg_plt_filename))
print(f"\n✨ Segmentation Loss Curve: {os.path.join(PLT_DIR, seg_plt_filename)}")

plt.figure(figsize=(12, 7))
plt.title('Mean Classification Loss Curves (Comparison)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Class Loss', fontsize=12)

for i, (model_name, data) in enumerate(all_model_data.items()):
    plt.plot(
        data['epochs'], 
        data['class_losses'], 
        marker=markers[i % len(markers)], 
        linestyle=linestyles[i % len(linestyles)], 
        color=colors[i % len(colors)], 
        label=model_name
    )

plt.xticks(range(1, max_epochs + 1))
plt.legend(title='Model')
plt.grid(True)
class_plt_filename = 'combined_class_loss.png'
plt.savefig(os.path.join(PLT_DIR, class_plt_filename))
print(f"✨ Classification Loss Curve: {os.path.join(PLT_DIR, class_plt_filename)}")