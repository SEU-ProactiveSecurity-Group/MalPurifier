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

# 导入Captum库，该库提供模型解释性工具，这里特别使用了集成梯度方法
from captum.attr import IntegratedGradients

# 导入NumPy
import numpy as np

# 从config模块中导入配置、日志和错误处理相关功能
from config import config, logging, ErrorHandler

# 导入自定义的工具模块
from tools import utils

# 初始化日志记录器并设置其名称
logger = logging.getLogger('core.defense.dt')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)

import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


class MalwareDetectionDT:
    def __init__(self, max_depth=None, random_state=0, name='DT', devide='cpu', **kwargs):
        """
        初始化恶意软件检测器

        参数:
        ----------
        @param max_depth: 树的最大深度。
        @param random_state: 整数，随机状态，用于重现结果。
        @param name: 字符串，用于命名模型。
        """
        
        del kwargs
        
        self.max_depth = max_depth
        self.random_state = random_state
        self.name = name
        self.n_classes = 2 # 二分类
        self.device = devide
        
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, 
                                            random_state=self.random_state)

        self.scaler = StandardScaler()  # 用于数据标准化
        
        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'dt') + '_' + self.name,
                                         'model.pkl')


    def forward(self, x):
        """
        通过随机森林模型获取预测的置信度
        
        参数
        ----------
        @param x: 2D张量或数组，特征表示
        
        返回值
        ----------
        返回预测的置信度
        """
        # 如果输入是PyTorch张量，我们需要将其转换为NumPy数组
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
            
        # Check if the input array is 1D and reshape
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # 标准化输入数据
        x = self.scaler.transform(x)

        # 使用随机森林模型得到预测的置信度
        confidences = self.model.predict_proba(x)

        # 如果需要将输出结果与PyTorch代码集成，我们可以将NumPy数组转换回PyTorch张量
        confidences = torch.tensor(confidences, dtype=torch.float32)

        return confidences

    def cross_validate(self, X, y, cv=5):
        """
        使用交叉验证评估模型性能。

        参数:
        ----------
        @param X: 训练数据。
        @param y: 目标标签。
        @param cv: 交叉验证的折数。
        
        返回:
        ----------
        scores: 交叉验证的分数列表。
        """
        # 首先对数据进行标准化
        X = self.scaler.fit_transform(X)
        
        # 执行交叉验证
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores
    
    
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
        confidences = []  # 存储每批数据的预测置信度
        gt_labels = []    # 存储每批数据的真实标签

        # 遍历每一批测试数据
        for x, y in test_data_producer:
            # 因为我们的模型已经被训练和标准化了，我们只需要转换数据为NumPy数组然后标准化
            x = self.scaler.transform(x.numpy())
            # 使用模型得到预测的置信度
            confidence_batch = self.model.predict_proba(x)
            confidences.append(confidence_batch)
            # 存储真实标签
            gt_labels.append(y.numpy())

        # 将所有批次的置信度垂直堆叠成一个数组
        confidences = np.vstack(confidences)
        # 将所有批次的真实标签连接成一个数组
        gt_labels = np.hstack(gt_labels)
        
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
        confidences = []  # 存储每批数据的预测置信度
        gt_labels = []    # 存储每批数据的真实标签

        # 遍历每一批测试数据
        for x, y in test_data_producer:
            # 因为我们的模型已经被训练和标准化了，我们只需要转换数据为NumPy数组然后标准化
            x = self.scaler.transform(x.numpy())
            # 使用模型得到预测的置信度
            confidence_batch = self.model.predict_proba(x)
            confidences.append(confidence_batch)
            # 存储真实标签
            gt_labels.append(y.numpy())

        return confidences, gt_labels
    
    from sklearn.metrics import accuracy_score



    def fit(self, train_data_producer, val_data_producer, early_stopping_rounds=30, n_resamples=None):
        # Load validation data first
        all_X_val = []
        all_y_val = []

        for batch_data in tqdm(val_data_producer, desc="Loading validation data"):
            X_batch, y_batch = batch_data
            all_X_val.append(X_batch.numpy())
            all_y_val.append(y_batch.numpy())

        X_val = np.vstack(all_X_val)
        y_val = np.hstack(all_y_val)

        # Standardize validation data
        X_val = self.scaler.fit_transform(X_val)

        best_val_accuracy = 0
        no_improve_rounds = 0  # Count the rounds without improvement in validation accuracy
        
        n_samples_per_batch = n_resamples or len(y_val)  # Default to validation set size

        # Start training
        for epoch, batch_data in enumerate(tqdm(train_data_producer, desc="Training batches")):
            X_batch, y_batch = batch_data
            X_batch = X_batch.numpy()
            y_batch = y_batch.numpy()

            # Sample with replacement to create a new batch
            indices = np.random.choice(len(y_batch), n_samples_per_batch, replace=True)
            X_resampled = X_batch[indices]
            y_resampled = y_batch[indices]

            # Standardize resampled batch
            X_resampled = self.scaler.transform(X_resampled)

            self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            self.model.fit(X_resampled, y_resampled)

            # Validate
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            # If validation accuracy improves, save the model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()
                no_improve_rounds = 0  # Reset the no improve counter
            else:
                no_improve_rounds += 1

            # If accuracy hasn't improved for early_stopping_rounds rounds, stop training
            if no_improve_rounds >= early_stopping_rounds:
                logger.info(f"Early stopping triggered as accuracy hasn't improved for {early_stopping_rounds} rounds.")
                break

        logger.info(f'{self.name} model trained and validated with best accuracy: {best_val_accuracy}')




        
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

        # 将张量转换为NumPy数组
        x = x.detach().cpu().numpy()

        # 获得模型的输出
        confidences = self.forward(x)

        # 返回每个样本的置信度和一个与logit形状相同的全1数组（表示恶意软件样本）
        return confidences, np.ones((confidences.size()[0],))
    
    
    def predict(self, test_data_producer, indicator_masking=True):
        """
        预测标签并进行评估

        参数
        --------
        @param test_data_producer: torch.DataLoader, 用于生成测试数据的数据加载器
        """
        # 进行评估
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1)  # 预测标签

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
        
        
    def save_model(self):
        """
        保存当前模型到磁盘。
        """
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        
        with open(self.model_save_path, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)
        logger.info(f'Model saved to {self.model_save_path}')

    def eval(self):
        pass

    def load(self):
        """
        从磁盘加载模型。
        """
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, 'rb') as file:
                saved = pickle.load(file)
                self.model = saved['model']
                self.scaler = saved['scaler']
            logger.info(f'Model loaded from {self.model_save_path}')
        else:
            logger.error(f'Model file not found at {self.model_save_path}')


    def load_state_dict(self):
        self.load()
           