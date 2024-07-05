from __future__ import absolute_import
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import os
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
logger = logging.getLogger('core.defense.rf')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)


class MalwareDetectionRF:
    def __init__(self, n_estimators=100, device='cpu', max_depth=None, random_state=0, name='RF', **kwargs):
        """
        初始化恶意软件检测器

        参数:
        ----------
        @param n_estimators: 树的数量。 
        @param max_depth: 树的最大深度。 
        @param random_state: 整数，随机状态，用于重现结果。
        @param name: 字符串，用于命名模型。
        """

        del kwargs

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.name = name
        self.device = device          # 定义运行设备
        self.n_classes = 2            # 二分类

        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state,
                                            n_jobs=-1)

        self.scaler = StandardScaler()  # 用于数据标准化

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'rf') + '_' + self.name,
                                         'model.pkl')

    def eval(self):
        pass

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

    def fit(self, train_data_producer):
        # 初始化列表，以便稍后将数据转换为NumPy数组
        all_X_train = []
        all_y_train = []

        # 使用 tqdm 包裹数据加载器以显示进度条
        for batch_data in tqdm(train_data_producer, desc="Loading data"):
            X_batch, y_batch = batch_data
            all_X_train.append(X_batch.numpy())
            all_y_train.append(y_batch.numpy())

        # 将列表转换为NumPy数组
        X_train = np.vstack(all_X_train)
        y_train = np.hstack(all_y_train)

        print("X_train.shape:", X_train.shape)

        # 标准化数据
        X_train = self.scaler.fit_transform(X_train)

        # 开始训练
        start_time = time.time()

        with tqdm(total=1, desc="Training RandomForest") as pbar:
            self.model.fit(X_train[:5000], y_train[:5000])
            pbar.update(1)

        end_time = time.time()

        training_time = end_time - start_time
        logger.info(
            f'{self.name} model trained successfully in {training_time} seconds.')

        self.save_model()

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
