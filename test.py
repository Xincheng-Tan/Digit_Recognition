import joblib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from models import LeNet5, LeNet5Half, LeNet5_Dropout_0_01, LeNet5_Dropout_0_05, LeNet5_Dropout_0_1

save_dir = "./test/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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


num_classes = 10
batch_size = 256
MODEL_CONFIGS = [
    ("LeNet5", LeNet5),
    ("LeNet5Half", LeNet5Half),
    ("LeNet5_Dropout_0_01", LeNet5_Dropout_0_01),
    ("LeNet5_Dropout_0_05", LeNet5_Dropout_0_05),
    ("LeNet5_Dropout_0_1", LeNet5_Dropout_0_1),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


def plot_confusion_matrix(cm, labels_name, title, colorbar=False):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title)
    
    if colorbar:
        plt.colorbar()
        
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', 
                         color="white" if cm[j, i] > thresh else "black") 
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    # plt.show()
    

def evaluate(predictions, labels, model_name="Model"):
    print("-" * 20)
    print(f"--- {model_name} ---")
    f_accuracy = accuracy_score(labels, predictions)
    print("Accuracy on test set: {:.2f}%".format(
        100. * f_accuracy
    ))
    
    # confusion_matrix
    f_matrix = confusion_matrix(labels, predictions) 
    plot_confusion_matrix(f_matrix, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], f"{model_name} Confusion Matrix")

    f_precise = np.zeros(10)
    f_recall = np.zeros(10)
    f_F1 = np.zeros(10)
    
    # P / R / F1 
    row_sum = np.sum(f_matrix, axis=1)
    col_sum = np.sum(f_matrix, axis=0)
    for i in range(10):
        f_precise[i] = f_matrix[i, i] / (col_sum[i] if col_sum[i] != 0 else 1e-8)
        f_recall[i] = f_matrix[i, i] / (row_sum[i] if row_sum[i] != 0 else 1e-8)
        # F1 Score
        if f_precise[i] + f_recall[i] != 0:
            f_F1[i] = 2 * f_precise[i] * f_recall[i] / (f_precise[i] + f_recall[i])
        else:
            f_F1[i] = 0.0
            
        print("Class {} :\tPrecision: {:.2f}%\tRecall: {:.2f}%\tF1: {:.4f}".format(
            i, 100. * f_precise[i], 100. * f_recall[i], f_F1[i]
        ))
        
    f_macro_precise = np.mean(f_precise)
    f_macro_recall = np.mean(f_recall)
    f_macro_F1 = np.mean(f_F1)
    
    print("Macro Precision:{:.2f}%\tMacro Recall:{:.2f}%\tMacro F1:{:.4f}".format(
        100. * f_macro_precise, 100. * f_macro_recall, f_macro_F1
    ))
    print("-" * 20)


def evaluate_pytorch_model(model_name, model_class, num_classes):
    net = model_class(num_classes)
    net.load_state_dict(torch.load(f"./weights/{model_name}.pt"))
    net = net.to(device)
    
    predict_y, true_y = [], []
    net.eval()
    with torch.no_grad():
        for test_data, test_target in test_loader:
            test_data = test_data.to(device)
            test_target = test_target.to(device)
            output = net(test_data)
            predicts = torch.argmax(output, dim=1)
            
            predict_y.extend(predicts.cpu().tolist())
            true_y.extend(test_target.cpu().tolist())
            
    evaluate(np.array(predict_y), np.array(true_y), model_name=model_name)


sys.stdout = Logger('test.log', sys.stdout)

try:
    for name, model_class in MODEL_CONFIGS:
        evaluate_pytorch_model(name, model_class, num_classes)

    # 将数据集转换为 NumPy 数组
    test_x = []
    test_y = []
    for x, y in test_dataset:
        test_x.append(x.numpy().flatten())
        test_y.append(y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # 数据形状调整
    test_x = test_x.reshape(len(test_y), 28*28)

    # 逻辑回归
    model_logic = joblib.load('./weights/logic.pkl')
    predict_y = model_logic.predict(test_x)
    evaluate(predict_y, test_y, model_name="Logistic Regression")

    # 决策树
    model_tree = joblib.load('./weights/tree.pkl')
    predict_y = model_tree.predict(test_x)
    evaluate(predict_y, test_y, model_name="Decision Tree")

    # 随机森林（基模型采用决策树）
    num_tree = 21
    models_tree = []
    for i in range(num_tree):
        model = joblib.load(f'./weights/RF/tree_{i+1}.pkl')
        models_tree.append(model)
    predict_arr = np.zeros(shape=(len(test_y), num_tree), dtype=np.int64)
    # 遍历决策树模型
    for j, tree_model in enumerate(models_tree):
        predict_arr[:, j] = tree_model.predict(test_x)
    # 投票, 出现频率最高的作为预测值
    predict_y = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predict_arr)
    evaluate(predict_y, test_y, model_name="Random Forest")

    # 支持向量机
    model_svm = joblib.load('./weights/SVM.pkl')
    predict_y = model_svm.predict(test_x)
    evaluate(predict_y, test_y, model_name="SVM")

except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    raise

finally:
    if isinstance(sys.stdout, Logger):
        sys.stdout.log.close()
