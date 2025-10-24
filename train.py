import os
import sys
import joblib
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models import LeNet5, LeNet5Half, LeNet5_Dropout_0_01, LeNet5_Dropout_0_05, LeNet5_Dropout_0_1
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm

save_dir = "./train/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

input_shape = 28
num_classes = 10
batch_size = 256
num_epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()
MODEL_CONFIGS = [
    ("LeNet5", LeNet5),
    ("LeNet5Half", LeNet5Half),
    ("LeNet5_Dropout_0_01", LeNet5_Dropout_0_01),
    ("LeNet5_Dropout_0_05", LeNet5_Dropout_0_05),
    ("LeNet5_Dropout_0_1", LeNet5_Dropout_0_1),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
# 将数据集分批次，丢弃不完整 Batch
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class Logger(object):
    def __init__(self, filename="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(os.path.join(save_dir, f'{filename}'), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        self.log.flush() 

    def flush(self):
        pass
        # self.terminal.flush()


def evaluate_model(net, test_loader, criterion, device, batch_size):
    """ Calculate the loss and accuracy of the model on the test set """
    net.eval()
    right_num, test_loss_sum = 0, 0
    with torch.no_grad():
        for test_data, test_target in test_loader:
            test_data = test_data.to(device)
            test_target = test_target.to(device)
            output = net(test_data)
            loss_test = criterion(output, test_target)
            predicts = torch.argmax(output, dim=1)
            test_loss_sum += loss_test.item()
            right_num += torch.sum(predicts == test_target).item()

    # 计算测试集样本总数（drop_last=True）
    total_test_samples_processed = batch_size * len(test_loader)
    
    epoch_test_loss = test_loss_sum / len(test_loader) 
    epoch_test_acc = right_num / total_test_samples_processed
    
    return epoch_test_loss, epoch_test_acc, total_test_samples_processed


sys.stdout = Logger('train.log', sys.stdout)
try:
    all_histories_loss = {}
    all_histories_acc = {}

    colors = plt.cm.get_cmap('tab10', len(MODEL_CONFIGS))

    for model_name, ModelClass in MODEL_CONFIGS:
        print("-" * 20)
        print(f"Starting training for model: {model_name}")

        net = ModelClass(num_classes=num_classes)
        net = net.to(device)
        print(net)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        history = {'loss_test': [], 'acc_test': []}
        
        initial_test_loss, initial_test_acc, total_test_samples_processed = evaluate_model(
            net, test_loader, criterion, device, batch_size
        )
        history['loss_test'].append(initial_test_loss)
        history['acc_test'].append(initial_test_acc)
        print(f"Initial (Epoch 0) Test loss: {initial_test_loss:.6f}\tInitial Test Acc: {initial_test_acc:.4f}")
        
        print(f"Training {model_name} for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            # 训练阶段
            net.train()
            train_loss_sum = 0.0
            for batch_idx, (train_data, train_target) in enumerate(train_loader):
                train_data = train_data.to(device)
                train_target = train_target.to(device)
                
                optimizer.zero_grad()
                train_output = net(train_data)
                loss = criterion(train_output, train_target)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() # 累加训练损失
            
            epoch_test_loss, epoch_test_acc, _ = evaluate_model(
                net, test_loader, criterion, device, batch_size
            )

            history['loss_test'].append(epoch_test_loss)
            history['acc_test'].append(epoch_test_acc)

            print("Model: {}\tepoch: {}\tTrain loss: {:.6f}\tTest loss: {:.6f}\tTest Acc: {:.4f}".format(
                model_name, epoch + 1, train_loss_sum / len(train_loader), epoch_test_loss, epoch_test_acc
            ))

        save_path = os.path.join(weights_dir, f'{model_name}.pt')
        torch.save(net.state_dict(), save_path)
        print(f"Model weights saved to: {save_path}")

        all_histories_loss[model_name] = history['loss_test']
        all_histories_acc[model_name] = history['acc_test']

    print("All models trained. Generating plots...")

    plotting_epochs_range = range(1, num_epochs + 1)

    # 测试集损失对比
    plt.figure(figsize=(10, 6))
    for i, (model_name, loss_history) in enumerate(all_histories_loss.items()):
        plt.plot(plotting_epochs_range, loss_history[1:], label=model_name, color=colors(i), marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Comparison of Loss between Different Versions of LeNet5 on the Test Set")
    plt.legend(loc='best')
    plt.xticks(plotting_epochs_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_path = os.path.join(save_dir, 'model_comparison_test_loss.png')
    plt.savefig(save_path)
    plt.show()

    # 测试集准确率对比
    plt.figure(figsize=(10, 6))
    for i, (model_name, acc_history) in enumerate(all_histories_acc.items()):
        plt.plot(plotting_epochs_range, acc_history[1:], label=model_name, color=colors(i), marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Accuracy of Different Versions of LeNet5 on the Test Set")
    plt.legend(loc='best')
    plt.xticks(plotting_epochs_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_path = os.path.join(save_dir, 'model_comparison_test_accuracy.png')
    plt.savefig(save_path)
    plt.show()

    print("Plots saved as 'model_comparison_test_loss.png' and 'model_comparison_test_accuracy.png'.")
    
    # 将数据集转换为 NumPy 数组
    train_x = []
    train_y = []
    for x, y in train_dataset:
        train_x.append(x.numpy().flatten())
        train_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_x = train_x.reshape(len(train_y), 28*28)
    
    # 逻辑回归
    print("\n -- Logic --")
    model_logic = LogisticRegression(random_state=1, multi_class='multinomial', max_iter=2000)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model_logic.fit(train_x, train_y)
    save_path = os.path.join(weights_dir, 'logic.pkl')
    joblib.dump(model_logic, save_path)
    '''
    '''
    # 决策树
    print("\n -- Tree --")
    model_tree = tree.DecisionTreeClassifier(random_state=1, criterion='entropy')
    model_tree.fit(train_x, train_y)
    save_path = os.path.join(weights_dir, 'tree.pkl')
    save_path = "./weights/tree.pkl"
    joblib.dump(model_tree, save_path)
    '''
    '''
    # 随机森林（基模型采用决策树）
    print("\n -- RF --")
    num_tree = 21
    models_tree = []
    for i in range(num_tree):
        index = np.random.choice(len(train_y), len(train_y), replace=True)
        # 信息熵作为划分属性指标, max_features 为子属性集大小
        model = tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_features=50)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model.fit(train_x[index], train_y[index])
        models_tree.append(model)
    rf_dir = os.path.join(weights_dir, 'RF/')
    if not os.path.exists(rf_dir):
        os.makedirs(rf_dir, exist_ok=True)
    for i in range(num_tree):
        save_path = os.path.join(rf_dir, f'tree_{i+1}.pkl')
        joblib.dump(models_tree[i], save_path)
    '''
    '''
    # 支持向量机
    print("\n -- SVM --")
    model_svm = svm.SVC()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model_svm.fit(train_x, train_y)
    save_path = os.path.join(weights_dir, 'SVM.pkl')
    joblib.dump(model_svm, save_path)

except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    raise

finally:
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close()