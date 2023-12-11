# 使用未来版本特性，确保代码在Python2和Python3中有一致的行为
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入基础库
import time
import os.path as path

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

# 导入Captum库，该库提供模型解释性工具，这里特别使用了集成梯度方法
from captum.attr import IntegratedGradients

# 导入NumPy
import numpy as np

# 从config模块中导入配置、日志和错误处理相关功能
from config import config, logging, ErrorHandler

# 导入自定义的工具模块
from tools import utils

# 初始化日志记录器并设置其名称
logger = logging.getLogger('core.defense.svm')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)

class MalwareDetectionSVM(nn.Module):
    """
    Using fully connected neural network to implement linear SVM and Logistic regression with hinge loss and
    cross-entropy loss which computes softmax internally, respectively.
    """
    def __init__(self, input_size, n_classes=2, device='cpu', name='md_svm', **kwargs):
        super(MalwareDetectionSVM, self).__init__()    # Call the init function of nn.Module
        self.input_size = input_size  # 定义输入尺寸
        self.n_classes = n_classes    # 定义分类数量
        self.device = device          # 定义运行设备
        self.name = name              # 定义模型名称
        
        self.fc = nn.Linear(self.input_size, self.n_classes)
        
        self.parse_args(**kwargs)   # 解析额外参数
        
        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'md_svm') + '_' + self.name,
                                         'model.pth')
        
        # 日志中打印模型的结构信息
        logger.info('========================================svm model architecture===============================')
        logger.info(self)
        logger.info('===============================================end==========================================')
        
    def parse_args(self,
                **kwargs
                ):
        """
        解析并设置网络的超参数。
        """
        # 从kwargs中获取并设置proc_number
        self.proc_number = kwargs.get('proc_number', None)  # 如果不存在，则返回None

        # 如果还有其他参数，记录警告，因为这些参数可能是未知的
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))


    def forward(self, x):
        out = self.fc(x)
        
        # 使用sigmoid来获取正类的概率
        return torch.sigmoid(out).squeeze()


    def inference(self, test_data_producer):
        """
        进行模型推理，获得预测的置信度和真实标签
        
        参数
        ----------
        @param test_data_producer: 数据生产者或数据加载器，用于产生测试数据
        
        返回值
        ----------
        返回预测的置信度和真实标签
        """
        confidences = []    # 存储每批数据的预测置信度
        gt_labels = []      # 存储每批数据的真实标签
        self.eval()         # 设置模型为评估模式

        # 使用torch.no_grad()来告诉PyTorch不要在推理过程中计算梯度
        with torch.no_grad():
            # 遍历每一批测试数据
            for x, y in test_data_producer:
                # 将数据转移到指定的设备（CPU或GPU）并调整数据类型
                x, y = utils.to_device(x.double(), y.long(), self.device)
                # 得到每一批数据的logits
                logits = self.forward(x)
                # 使用softmax函数得到每一批数据的置信度，并将其添加到confidences列表中
                confidences.append(F.softmax(logits, dim=-1))
                # 将每一批数据的真实标签添加到gt_labels列表中
                gt_labels.append(y)

        # 将所有批次的置信度垂直堆叠成一个张量
        confidences = torch.vstack(confidences)
        # 将所有批次的真实标签连接成一个张量
        gt_labels = torch.cat(gt_labels, dim=0)
        
        return confidences, gt_labels


    def inference_dae(self, test_data_producer):
        """
        进行模型推理，获得预测的置信度和真实标签
        
        参数
        ----------
        @param test_data_producer: 数据生产者或数据加载器，用于产生测试数据
        
        返回值
        ----------
        返回预测的置信度和真实标签
        """
        confidences = []    # 存储每批数据的预测置信度
        gt_labels = []      # 存储每批数据的真实标签
        self.eval()         # 设置模型为评估模式

        # 使用torch.no_grad()来告诉PyTorch不要在推理过程中计算梯度
        with torch.no_grad():
            # 遍历每一批测试数据
            for x, y in test_data_producer:
                # 将数据转移到指定的设备（CPU或GPU）并调整数据类型
                x, y = utils.to_device(x.double(), y.long(), self.device)
                # 得到每一批数据的logits
                logits = self.forward(x)
                # 使用softmax函数得到每一批数据的置信度，并将其添加到confidences列表中
                confidences.append(F.softmax(logits, dim=-1))
                # 将每一批数据的真实标签添加到gt_labels列表中
                gt_labels.append(y)
        
        return confidences, gt_labels


    def inference_batch_wise(self, x):
        """
        仅支持恶意软件样本的批量推理
        
        参数
        ----------
        @param x: 输入数据的张量
        
        返回值
        ----------
        返回推理的置信度和标签
        """
        # 确保输入是一个张量
        assert isinstance(x, torch.Tensor)
        
        # 获得模型的输出
        logit = self.forward(x)
        
        # 返回每个样本的置信度和一个与logit形状相同的全1数组（表示恶意软件样本）
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))


    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        训练SVM模型，并根据验证集上的损失选择最佳模型。
        """
        # 初始化优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MultiMarginLoss()

        best_avg_acc = 0.   # 记录验证集上的最佳准确率
        best_epoch = 0      # 记录最佳准确率对应的周期

        # 获取训练数据批次的数量
        nbatches = len(train_data_producer)
        
        for i in range(epochs):
            self.train()
            running_corrects = 0

            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train = x_train.double().to(self.device)
                y_train = y_train.long().to(self.device)
                
                optimizer.zero_grad()        
                                    
                outputs = self.forward(x_train)  
                loss = criterion(outputs, y_train)
                
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == y_train).item()

                if verbose and (idx_batch % 10 == 0): # 打印每10批次的训练情况
                    print(f"Epoch {i}/{epochs}, Batch {idx_batch}/{nbatches} - Loss: {loss.item():.4f}")

            epoch_acc = running_corrects / len(train_data_producer.dataset)
            
            self.eval()  # 将模型设置为评估模式
            val_corrects = 0
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val = x_val.double().to(self.device)
                    y_val = y_val.long().to(self.device)

                    outputs = self.forward(x_val)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == y_val).item()

            val_acc = val_corrects / len(validation_data_producer.dataset)

            if val_acc > best_avg_acc:
                best_avg_acc = val_acc
                best_epoch = i
                
                # 检查模型保存路径是否存在，如果不存在，则创建
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))
                
                # 保存当前的模型参数
                torch.save(self.state_dict(), self.model_save_path)
                
                # 如果开启了详细输出模式，显示模型保存路径
                if verbose:
                    print(f'模型保存在路径： {self.model_save_path}')

            if verbose:
                print(f"Epoch {i}/{epochs} - Training Accuracy: {epoch_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        print(f"Best Validation Accuracy: {best_avg_acc:.4f} at Epoch {best_epoch}")

    def predict(self, test_data_producer, indicator_masking=True):
        """
        预测标签并进行评估

        参数
        --------
        @param test_data_producer: torch.DataLoader, 用于生成测试数据的数据加载器
        """
        # 进行评估
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1).cpu().numpy()  # 预测标签
        y_true = y_true.cpu().numpy()                # 真实标签
        
        # 使用sklearn的评估指标进行评估
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        MSG = "The accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(accuracy * 100))
        
        MSG = "The balanced accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(b_accuracy * 100))

        # 检查数据中是否存在缺失的类别
        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)                        # 计算假阳性率
        fnr = fn / float(tp + fn)                        # 计算假阴性率
        f1 = f1_score(y_true, y_pred, average='binary')  # 计算F1分数

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%、False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))    

    def load(self):
        """
        从磁盘加载模型参数
        """
        self.load_state_dict(torch.load(self.model_save_path))
