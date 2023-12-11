from os import path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_kde')
logger.addHandler(ErrorHandler)

# 使用核密度估计在倒数第二层进行异常检测的类
class KernelDensityEstimation(DetectorTemplate):
    """
    核密度估计在倒数第二层

    参数
    -------------
    @param model, torch.nn.Module, 一个模型对象实例
    @param bandwidth, float, 高斯密度函数的方差
    @param n_classes, 整数, 类别数量
    @param ratio, float [0,1], 计算阈值的比例
    """

    def __init__(self, model, n_centers=1000, bandwidth=20., n_classes=2, ratio=0.9):
        super(KernelDensityEstimation, self).__init__()     # 调用父类的初始化函数
        assert isinstance(model, torch.nn.Module)           # 确保提供的模型是torch.nn.Module类型
        
        self.model = model          # 设置模型
        self.device = model.device  # 设定设备
        self.n_centers = n_centers  # 设置中心点数量
        self.bandwidth = bandwidth  # 设置带宽
        self.n_classes = n_classes  # 设置类别数量
        self.ratio = ratio          # 设置比例
        self.gaussian_means = None  # 初始化高斯均值为None

        # 初始化tau为一个不需要梯度的参数，大小为[n_classes,]
        self.tau = nn.Parameter(torch.zeros([self.n_classes, ], device=self.device), requires_grad=False)
        self.name = self.model.name  # 设置名称为模型的名称
        self.model.load()            # 加载模型
        
        # 设置模型保存路径
        self.model_save_path = path.join(config.get('experiments', 'amd_kde').rstrip('/') + '_' + self.name, 'model.pth')
        self.model.model_save_path = self.model_save_path


    def forward(self, x):
        # 对于模型中的密集层除了最后一个，都进行前向传播，并使用激活函数
        for dense_layer in self.model.dense_layers[:-1]:
            x = self.model.activation_func(dense_layer(x))
            
        # 获取最后一层的输出
        logits = self.model.dense_layers[-1](x) 
        
        # print("logits.shape:", logits.shape)
                
        if len(logits.shape) == 1:
            x_prob = self.forward_g(x, logits.argmax().detach())    # 适用于形状为[2]的logits
        else:
            x_prob = self.forward_g(x, logits.argmax(1).detach())   # 适用于形状为[10, 2]的logits
            
        return logits, x_prob


    def forward_f(self, x):
        # 对于模型中的密集层除了最后一个，都进行前向传播，并使用激活函数
        for dense_layer in self.model.dense_layers[:-1]:
            x = self.model.activation_func(dense_layer(x))
        logits = self.model.dense_layers[-1](x)  # 获取最后一层的输出
        return logits, x


    # 这里，我们计算了每个样本和gaussian_means（高斯均值）之间的距离，
    # 然后使用核密度估计计算其概率。最后，我们返回与预测标签对应的密度值的负数。
    def forward_g(self, x_hidden, y_pred):
        """
        根据核密度估计对隐藏层表示计算其概率。

        参数
        -----------
        @param x_hidden, torch.tensor, 节点的隐藏表示
        @param y_pred, torch.tensor, 预测结果
        """
        
        # print("x_hidden shape:", x_hidden.shape)
        # print("size = x_hidden.size()", x_hidden.size())

        # 如果x_hidden只有一个维度, 我们增加一个新的维度来符合后面的操作
        if len(x_hidden.shape) == 1:
            x_hidden = x_hidden.unsqueeze(0)

        size = x_hidden.size()[0]  # 获取x_hidden的第一维度大小，这应该是批次大小

        # 计算每个样本与高斯均值之间的距离，并平方
        dist = [torch.sum(torch.square(means.unsqueeze(dim=0) - x_hidden.unsqueeze(dim=1)), dim=-1) for means in
                self.gaussian_means]

        # 对计算得到的距离进行核密度估计
        kd = torch.stack([torch.mean(torch.exp(-d / self.bandwidth ** 2), dim=-1) for d in dist], dim=1)

        # 返回与预测标签对应的密度的负值
        return -1 * kd[torch.arange(size), y_pred]



    def get_threshold(self, validation_data_producer, ratio=None):
        """
        获取核密度估计的阈值
        :@param validation_data_producer: 对象，用于生成验证数据集的迭代器
        :@param ratio: 阈值计算所使用的比例
        """

        # 如果提供了比例参数，则使用提供的值，否则使用类初始化时设定的比例值
        ratio = ratio if ratio is not None else self.ratio
        assert 0 <= ratio <= 1  # 确保比例在0到1之间

        self.eval()         # 将模型设置为评估模式（不使用dropout等）
        probabilities = []  # 用于存储概率的列表
        gt_labels = []      # 用于存储真实标签的列表

        # 确保不计算梯度（提高计算效率，减少内存使用）
        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                # 将数据转为tensor，并移至指定设备（例如GPU）
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.device)
                logits, x_prob = self.forward(x_val)    # 使用模型进行前向传播
                probabilities.append(x_prob)            # 将概率值添加到列表中
                gt_labels.append(y_val)                 # 将真实标签添加到列表中

            # 将所有概率和标签的列表连接成一个大的tensor
            prob = torch.cat(probabilities, dim=0)
            gt_labels = torch.cat(gt_labels)

            # 对于每个类别，找出该类别下的密度值，并根据比例找出对应的阈值
            for i in range(self.n_classes):
                prob_x_y = prob[gt_labels == i]
                s, _ = torch.sort(prob_x_y)                     # 对密度值进行排序
                self.tau[i] = s[int((s.shape[0] - 1) * ratio)]  # 根据比例选择阈值，并存入tau中


    def predict(self, test_data_producer, indicator_masking=True):
        """
        对测试数据进行预测并评估结果。
        :@param test_data_producer: 对象，用于生成测试数据的迭代器
        :@param indicator_masking: 布尔值，决定是否使用指示器进行筛选
        """

        # 在测试数据上进行推理，并获取中心预测、概率值和真实标签
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        
        # 获取最大概率的类别作为预测标签
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        
        # 使用指示器进行筛选
        indicator_flag = self.indicator(x_prob, y_pred).cpu().numpy()
        if indicator_masking:
            # 如果开启指示器筛选，则只保留"确定"的样本
            flag_of_retaining = indicator_flag
            y_pred = y_pred[flag_of_retaining]
            y_true = y_true[flag_of_retaining]
        else:
            # 否则，将"不确定"的样本预测为类别1
            y_pred[~indicator_flag] = 1.

        logger.info('The indicator is turning on...')
        
        # 从sklearn库中导入多个评估指标
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        
        # 计算准确率和平衡准确率
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)
        logger.info("The accuracy on the test dataset is {:.5f}%".format(accuracy * 100))
        logger.info("The balanced accuracy on the test dataset is {:.5f}%".format(b_accuracy * 100))

        # 检查是否有缺失的类别
        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        # 计算混淆矩阵，并从中获取真正例、假正例、真负例和假负例的数量
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)  # 假正例率
        fnr = fn / float(tp + fn)  # 假负例率
        f1 = f1_score(y_true, y_pred, average='binary')  # F1分数

        print("Other evaluation metrics we may need:")
        logger.info("False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
                    .format(fnr * 100, fpr * 100, f1 * 100))


    def eval(self):
        self.model.eval()

    def inference(self, test_data_producer):
        """
        对给定的测试数据进行推理，获取类别中心、概率值和真实标签。
        :@param test_data_producer: 对象，用于生成测试数据的迭代器
        :return: 类别中心、概率值和真实标签
        """

        y_cent, x_prob = [], []  # 用于存储类别中心和概率值的列表
        gt_labels = []  # 用于存储真实标签的列表
        
        # 将模型设置为评估模式，确保在推理时不进行权重更新或其他训练特定的操作
        self.eval()
        
        # 不进行梯度计算
        with torch.no_grad():
            for x, y in test_data_producer:
                # 转换数据类型并将其移动到正确的设备（例如，GPU）
                x, y = utils.to_tensor(x.double(), y.long(), self.device)
                
                # 使用模型进行前向传播
                logits_f, logits_g = self.forward(x)
                
                # 将结果添加到列表中
                y_cent.append(F.softmax(logits_f, dim=-1))  # 使用softmax函数计算类别中心
                x_prob.append(logits_g)  # 将概率值添加到列表
                gt_labels.append(y)  # 添加真实标签

        # 将列表内容连接成张量
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        
        return y_cent, x_prob, gt_labels

    def inference_batch_wise(self, x):
        """
        批量推理函数。对给定的输入张量进行推理，并返回softmax后的输出和概率值。
        :@param x: torch.Tensor，输入数据张量
        :return: softmax后的输出和概率值
        """

        # 确保输入是一个torch张量
        assert isinstance(x, torch.Tensor)

        # 使用模型进行前向传播
        logits, x_prob = self.forward(x)
        
        # 返回softmax处理后的输出和概率值
        return torch.softmax(logits, dim=-1).detach().cpu().numpy(), x_prob.detach().cpu().numpy()


    def get_tau_sample_wise(self, y_pred=None):
        return self.tau[y_pred]

    def indicator(self, x_prob, y_pred=None):
        """
        指示器函数，根据概率值x_prob和预测标签y_pred，判断概率值是否小于等于给定样本的tau值。
        :@param x_prob: 概率值，可以是numpy数组或torch张量。
        :@param y_pred: 预测标签，应为非空。
        :return: 布尔数组或张量，表示x_prob中的每个值是否小于等于相应的tau值。
        """
        
        # 确保提供了预测标签
        assert y_pred is not None
        
        # 如果x_prob是numpy数组
        if isinstance(x_prob, np.ndarray):
            # 转换x_prob为torch张量
            x_prob = torch.tensor(x_prob, device=self.device)
            # 检查x_prob中的每个值是否小于等于对应的tau值，并将结果返回为numpy数组
            return (x_prob <= self.get_tau_sample_wise(y_pred)).cpu().numpy()
        
        # 如果x_prob是torch张量
        elif isinstance(x_prob, torch.Tensor):
            # 检查x_prob中的每个值是否小于等于对应的tau值，并返回结果张量
            return x_prob <= self.get_tau_sample_wise(y_pred)
        
        else:
            # 如果x_prob既不是numpy数组也不是torch张量，则抛出类型错误
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # fit函数首先使用训练数据集生成器处理输入数据
    # 收集隐藏层的输出并为每个类别计算高斯均值。
    # 然后，它使用验证数据集生成器来计算和设置核密度估计的阈值，并将训练好的模型保存到磁盘。
    def fit(self, train_dataset_producer, val_dataset_producer):
        """
        根据给定的训练数据集训练并设置模型，同时为核密度估计确定阈值。
        
        :@param train_dataset_producer: 训练数据集的生成器。
        :@param val_dataset_producer: 验证数据集的生成器。
        """

        # 用于存储隐藏层的输出和真实标签的列表
        X_hidden, gt_labels = [], []
        
        # 设置模型为评估模式
        self.eval()

        # 确保不进行梯度计算
        with torch.no_grad():
            # 对于train_dataset_producer中的每一批数据
            for x, y in train_dataset_producer:
                # 将输入数据和标签转换为torch张量，并将其移动到模型所在的设备上
                x, y = utils.to_tensor(x.double(), y.long(), self.device)

                # 使用模型的forward_f函数处理输入数据，并获取输出结果
                logits, x_hidden = self.forward_f(x)

                # 将隐藏层的输出和真实标签添加到对应的列表中
                X_hidden.append(x_hidden)
                gt_labels.append(y)

                # 检查每个类别的样本数量，确保每个类别都有足够的样本进行后续的处理
                _, count = torch.unique(torch.cat(gt_labels), return_counts=True)
                if torch.min(count) >= self.n_centers:
                    break

            # 将收集的隐藏层输出和真实标签列表转换为torch张量
            X_hidden = torch.vstack(X_hidden)
            gt_labels = torch.cat(gt_labels)

            # 为每个类别计算高斯均值，并存储到self.gaussian_means中
            self.gaussian_means = [X_hidden[gt_labels == i][:self.n_centers] for i in range(self.n_classes)]

        # 使用验证数据集来计算和设置核密度估计的阈值
        self.get_threshold(val_dataset_producer)

        # 如果模型保存路径不存在，则创建相应的目录
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        
        # 将模型保存到磁盘
        self.save_to_disk()


    def load(self):
        """
        从磁盘加载保存的模型参数和其他重要数据。
        """
        ckpt = torch.load(self.model_save_path)         # 从给定的保存路径加载模型检查点
        self.gaussian_means = ckpt['gaussian_means']    # 从检查点中加载高斯均值
        self.tau = ckpt['tau']                          # 从检查点中加载阈值
        self.model.load_state_dict(ckpt['base_model'])  # 从检查点中加载基模型的参数

    def save_to_disk(self):
        """
        将模型参数和其他重要数据保存到磁盘。
        """
        torch.save({
            'gaussian_means': self.gaussian_means,      # 保存高斯均值
            'tau': self.tau,                            # 保存阈值
            'base_model': self.model.state_dict()       # 保存基模型的参数
        },
            self.model_save_path                        # 指定保存路径
        )
