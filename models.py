import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        # C1: 卷积层 (1, 28, 28) -> (6, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU() # 原为 Sigmoid/Tanh, 这里使用更好的 ReLU
        )
        # S2: 池化层 (6, 24, 24) -> (6, 12, 12)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C3: 卷积层 (6, 12, 12) -> (16, 8, 8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU()
        )
        # S4: 池化层 (16, 8, 8) -> (16, 4, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C5/F5: 全连接层/卷积层 (16, 4, 4) -> 256
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )
        # F6: 全连接层
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # Output: 全连接层
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output


class LeNet5Half(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5Half, self).__init__()        
        # 卷积核 6 -> 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积核 16 -> 8
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 4 * 4, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output


class LeNet5_Dropout_0_01(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5_Dropout_0_01, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(p=0.01) # 增加 1% dropout
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.01) # 增加 1% dropout
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
    

class LeNet5_Dropout_0_05(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5_Dropout_0_05, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(p=0.05) # 增加 5% dropout
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.05) # 增加 5% dropout
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
    

class LeNet5_Dropout_0_1(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5_Dropout_0_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(p=0.1) # 增加 10% dropout
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.1) # 增加 10% dropout
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output