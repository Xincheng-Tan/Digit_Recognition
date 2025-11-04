# 手写数字识别

## 1 Directory Structure
- **`./data/`**: 存放 MNIST 数据集。train.py 自动创建文件夹并下载
- **`./weights/`**: 存放模型的权重。train.py 自动保存训练好的权重文件
- **`./train/`**: 存放 train.py 的中间输出结果和日志
- **`./test/`**: 存放 test.py 的中间输出结果和日志
- **models.py**: 定义不同版本的 LeNet5 模型：
    1. **LeNet5**: 针对 MNIST 数据集修改的 LeNet5 网络
    2. **LeNet5Half**: 在 LeNet5 基础上，所有卷积层去掉一半的卷积核
    3. **LeNet5_Dropout_0_01**: 在 LeNet5 基础上，增加 1% dropout 层
    4. **LeNet5_Dropout_0_05**: 在 LeNet5 基础上，增加 5% dropout 层
    5. **LeNet5_Dropout_0_1**: 在 LeNet5 基础上，增加 10% dropout 层
- **train.py**: 总训练代码
- **eval.py**: 总测试代码
- **requirements.txt**: 库及版本

## 2 Installation
配置虚拟环境
```shell script
git clone https://github.com/Xincheng-Tan/Digit_Recognition.git
conda create -n digit python=3.10
conda activate digit
cd Digit_Recognition & pip install -r requirements.txt
```

## 3 Train
包括对以下模型的训练：
- LeNet5 不同模型版本
- 逻辑回归
- 决策树
- 随机森林（包含21棵决策树，采用信息熵作为划分属性指标，随机选取属性子集数目为50）
- SVM

你可以一次训练多个模型、也可以通过“注释”的方法进行单个模型的训练  
训练得到的模型权重被保存到 **`./weights/`** 文件夹下：
- **`./RF/`**：存放随机森林各树（T=21）的权重文件
- **LeNet5.pt ...** ：LeNet5 不同模型版本的权重文件
- **logic.kpl**：逻辑回归模型权重文件
- **SVM.kpl**：支持向量机分类器权重文件
- **tree.kpl**：决策树权重文件

以下文件被保存到 **`./train/`** 下：
- **train.log**: 训练日志
- **model_comparison_test_accuracy.png、model_comparison_test_loss.png**: 训练过程中，LeNet5 不同版本在**测试集上准确率和损失变化曲线**

## 4 Eval
包括对上述所有模型权重的测试
你可以一次测试多个模型、也可以通过“注释”的方法进行单个模型的测试
以下文件被保存到 **`./test/`** 下：
- **test.log**: 测试日志
- **Decision Tree Confusion Matrix.png ...** : 各模型在测试集上的**混淆矩阵**

## 5 Results
**不同分类模型在测试集上的整体指标**
|Classification|Accuracy|Macro Precision|Macro Recall|Macro F1|
|---|---|---|---|---|
|LeNet5|**98.85%**|**98.84%**|**98.84%**|**0.9884**|
|Logistic Regression|92.59%|92.49%|92.48%|0.9248|
|Decision Tree|88.58%|88.45%|88.43%|0.8844|
|Random Forest (T=21)|95.57%|95.53%|95.51%|0.9551|
|SVM|97.92%|97.92%|97.91%|0.9791|

**LeNet5 不同模型版本在测试集上的整体指标**
|LeNet5 Versions|Accuracy|Macro Precision|Macro Recall|Macro F1|
|---|---|---|---|---|
|LeNet5|98.85%|98.84%|98.84%|0.9884|
|LeNet5Half|97.88%|97.88%|97.85%|0.9786|
|LeNet5_Dropout_0_01|98.86%|98.85%|98.86%|0.9885|
|LeNet5_Dropout_0_05|**98.98%**|**98.99%**|98.65%|**0.9897**|
|LeNet5_Dropout_0_1|98.95%|98.96%|**98.94%**|0.9895|

训练过程中，LeNet5 不同版本在**测试集上准确率变化曲线**
<p align="center">
  <img alt="model_comparison_test_accuracy" src="./doc/model_comparison_test_accuracy.png" width="98%">
</p>

训练过程中，LeNet5 不同版本在**测试集上损失变化曲线**
<p align="center">
  <img alt="model_comparison_test_loss" src="./doc/model_comparison_test_loss.png" width="98%">
</p>

LeNet5_Dropout_0_05 在**测试集上的混淆矩阵**
<p align="center">
  <img alt="LeNet5_Dropout_0_05 Confusion Matrix" src="./doc/LeNet5_Dropout_0_05 Confusion Matrix.png" width="98%">
</p>