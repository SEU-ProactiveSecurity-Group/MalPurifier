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

# 导入Captum库，该库提供模型解释性工具，这里特别使用了集成梯度方法
from captum.attr import IntegratedGradients

# 导入NumPy
import numpy as np

# 从config模块中导入配置、日志和错误处理相关功能
from config import config, logging, ErrorHandler

# 导入自定义的工具模块
from tools import utils

# 初始化日志记录器并设置其名称
logger = logging.getLogger('core.defense.dnn')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)


# 继承PyTorch的nn.Module类，定义恶意软件检测的深度神经网络
class MalwareDetectionDNN(nn.Module):
    def __init__(self, input_size, n_classes, device='cpu', name='DNN', **kwargs):
        """
        初始化恶意软件检测器

        参数:
        ----------
        @param input_size: 整数，输入向量的维度数量。
        @param n_classes: 整数，表示分类的数量，例如二分类问题中n=2。
        @param device: 字符串，可以是'cpu'或'cuda'，表示模型应该在CPU还是GPU上运行。
        @param name: 字符串，用于命名模型。
        """
        super(MalwareDetectionDNN, self).__init__()  # 调用父类初始化
        self.input_size = input_size  # 定义输入尺寸
        self.n_classes = n_classes    # 定义分类数量
        self.device = device          # 定义运行设备
        self.name = name              # 定义模型名称

        self.parse_args(**kwargs)   # 解析额外参数

        self.dense_layers = []  # 初始化一个空的密集层列表
        
        # 检查是否至少有一个隐藏层
        if len(self.dense_hidden_units) >= 1:
            # 添加第一个密集层
            self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        else:
            # 如果没有隐藏层，抛出异常
            raise ValueError("Expect at least one hidden layer.")

        # 为每一对连续的隐藏单元添加一个密集层
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.dense_layers.append(nn.Linear(self.dense_hidden_units[i], 
                                               self.dense_hidden_units[i + 1]))
        
        # 添加最后一个连接到输出层的密集层
        self.dense_layers.append(nn.Linear(self.dense_hidden_units[-1], self.n_classes))
        
        # 将密集层添加到模型中以进行跟踪
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('nn_model_layer_{}'.format(idx_i), dense_layer)

        # 根据参数选择使用SELU或ReLU激活函数
        if self.smooth:
            self.activation_func = F.selu  # 使用SELU激活函数
        else:
            self.activation_func = F.relu  # 使用ReLU激活函数

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + self.name,
                                         'model.pth')
        
        # 日志中打印模型的结构信息
        logger.info('========================================dnn model architecture===============================')
        logger.info(self)
        logger.info('===============================================end==========================================')


    def parse_args(self,
                dense_hidden_units=None,
                dropout=0.6,
                alpha_=0.2,
                smooth=False,
                **kwargs
                ):
        """
        解析并设置网络的超参数。

        参数:
        ----------
        dense_hidden_units : list, 可选
            网络中每个隐藏层的单元数。如果没有指定，则默认为两个隐藏层，每层200个单元。
        dropout : float, 可选
            dropout正则化的比率，默认为0.6。
        alpha_ : float, 可选
            某些激活函数的参数，默认为0.2。
        smooth : bool, 可选
            是否使用平滑的激活函数，默认为False。
        **kwargs : dict
            其他超参数。
        """

        # 如果用户没有指定隐藏层，使用默认的配置
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
        # 如果用户指定了一个列表，使用它
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
        # 否则抛出一个异常
        else:
            raise TypeError("Expect a list of hidden units.")

        # 设置dropout, alpha和smooth参数
        self.dropout = dropout
        self.alpha_ = alpha_
        self.smooth = smooth

        # 从kwargs中获取并设置proc_number
        self.proc_number = kwargs.get('proc_number', None)  # 如果不存在，则返回None

        # 如果还有其他参数，记录警告，因为这些参数可能是未知的
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))


    def forward(self, x):
        """
        使输入数据 x 通过神经网络
        
        参数
        ----------
        @param x: 2D张量，特征表示
        """
        # 遍历神经网络的每一层，除了最后一层
        for dense_layer in self.dense_layers[:-1]:
            x = self.activation_func(dense_layer(x))  # 使用激活函数处理每一层的输出

        # 对处理过的数据进行 dropout 操作，用于防止过拟合
        latent_representation = F.dropout(x, self.dropout, training=self.training)
        
        # 用最后一层进行处理，得到logits（未归一化的预测或分类得分）
        logits = self.dense_layers[-1](latent_representation)
        return logits

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


    def get_important_attributes(self, test_data_producer, target_label=1):
        """
        使用集成梯度（Integrated Gradients）方法获取重要的属性/特征

        参数
        ----------
        @param test_data_producer: 数据生产者或数据加载器，用于产生测试数据
        @param target_label: 目标标签，默认为1
        
        返回值
        ----------
        返回重要的属性/特征
        """
        attributions = []   # 存储属性或特征的重要性得分
        gt_labels = []      # 存储真实标签

        # 定义一个使用集成梯度方法的包装器
        def _ig_wrapper(_x):
            logits = self.forward(_x)
            return F.softmax(logits, dim=-1)

        # 初始化集成梯度对象
        ig = IntegratedGradients(_ig_wrapper)

        # 遍历测试数据集
        for i, (x, y) in enumerate(test_data_producer):
            # 将数据和标签转移到指定的设备上
            x, y = utils.to_device(x.double(), y.long(), self.device)
            # 使x能够计算梯度
            x.requires_grad = True
            # 定义基线，用于集成梯度的计算
            baseline = torch.zeros_like(x, dtype=torch.double, device=self.device)
            # 计算属性的重要性
            attribution_bs = ig.attribute(x,
                                          baselines=baseline,
                                          target=target_label)
            # 将所有批次的属性垂直堆叠
            attribution = torch.hstack(attribution_bs)
            # 保存得到的属性重要性得分和真实标签
            attributions.append(attribution.clone().detach().cpu().numpy())
            gt_labels.append(y.clone().detach().cpu().numpy())
            # 将真实标签保存为.npy文件
            np.save('./labels', np.concatenate(gt_labels))
            
        return np.vstack(attributions)


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
        
        # print("y_true.shape:", y_true.shape)
        # print("y_pred.shape:", y_pred.shape)
        
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


    def customize_loss(self, logits, gt_labels, representation=None, mini_batch_idx=None):
        """
        自定义损失函数

        参数
        --------
        @param logits: Tensor, 模型的输出
        @param gt_labels: Tensor, 真实的标签
        @param representation: Tensor, 可选参数，表示特征表示
        @param mini_batch_idx: Int, 可选参数，表示小批次的索引
        
        返回值
        --------
        返回交叉熵损失
        """
        return F.cross_entropy(logits, gt_labels)


    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        训练恶意软件检测器，根据验证集上的交叉熵损失选择最佳模型。

        参数
        ----------
        @param train_data_producer: 对象, 用于生成一批训练数据的迭代器
        @param validation_data_producer: 对象, 用于生成验证数据的迭代器
        @param epochs: 整数, 训练的周期数
        @param lr: 浮点数, Adam优化器的学习率
        @param weight_decay: 浮点数, 惩罚因子
        @param verbose: 布尔值, 是否显示详细的日志
        """
        # 初始化优化器
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.   # 记录验证集上的最佳准确率
        best_epoch = 0      # 记录最佳准确率对应的周期
        total_time = 0.     # 总的训练时间

        # 获取训练数据批次的数量
        nbatches = len(train_data_producer)
        
        # 进行指定次数的训练周期
        for i in range(epochs):
            # 设置模型为训练模式
            self.train()
            # 初始化列表用于保存每批数据的损失值和准确率
            losses, accuracies = [], []

            # 对每个训练数据批次进行遍历
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # 将数据转移到指定的计算设备（例如GPU或CPU）
                x_train, y_train = utils.to_device(x_train.double(), y_train.long(), self.device)

                # 记录开始训练的时间
                start_time = time.time()

                # 清空之前累积的梯度
                optimizer.zero_grad()        
                                       
                # 对输入数据进行前向传播
                logits = self.forward(x_train)  
                                    
                # 根据模型的输出和真实标签计算损失
                loss_train = self.customize_loss(logits, y_train)   

                # 对损失进行反向传播
                loss_train.backward()
                
                # 使用优化器更新模型参数
                optimizer.step()

                # 计算训练这批数据所花费的总时间
                total_time += time.time() - start_time
                
                # 计算这批数据上的准确率
                acc_train = (logits.argmax(1) == y_train).sum().item() / x_train.size()[0]
                
                # 将时间转换为分钟和秒
                mins, secs = int(total_time / 60), int(total_time % 60)
                
                # 将这批数据的损失和准确率加入到列表中
                losses.append(loss_train.item())
                accuracies.append(acc_train)

                # 如果开启了详细输出模式，显示当前训练进度和这批数据上的损失和准确率
                if verbose:
                    logger.info(f'小批次： {i * nbatches + idx_batch + 1}/{epochs * nbatches} | 训练时间为 {mins:.0f} 分钟, {secs} 秒。')
                    logger.info(f'训练损失（小批次级别）: {losses[-1]:.4f} | 训练精度: {acc_train * 100:.2f}')


            self.eval()  # 将模型设置为评估模式
            avg_acc_val = []

            with torch.no_grad():  # 确保在评估模式下不进行梯度的计算
                for x_val, y_val in validation_data_producer:
                    # 将数据移动到指定设备（例如GPU或CPU）上，并确保数据的类型为双精度浮点数和长整型
                    x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)
                    
                    # 使用模型进行前向传播，得到输出结果
                    logits = self.forward(x_val)
                    
                    # 计算验证数据上的准确率
                    acc_val = (logits.argmax(1) == y_val).sum().item() / x_val.size()[0]
                    
                    # 保存每一批验证数据的准确率
                    avg_acc_val.append(acc_val)
                
                # 计算所有验证数据的平均准确率
                avg_acc_val = np.mean(avg_acc_val)

            # 如果当前周期的验证精度超过之前的最佳验证精度
            if avg_acc_val >= best_avg_acc:
                # 更新最佳验证精度
                best_avg_acc = avg_acc_val
                best_epoch = i
                
                # 检查模型保存路径是否存在，如果不存在，则创建
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))
                
                # 保存当前的模型参数
                torch.save(self.state_dict(), self.model_save_path)
                
                # 如果开启了详细输出模式，显示模型保存路径
                if verbose:
                    print(f'模型保存在路径： {self.model_save_path}')

            # 如果开启了详细输出模式，显示训练损失、训练精度、验证精度和最佳验证精度
            if verbose:
                logger.info(f'训练损失（周期级别）: {np.mean(losses):.4f} | 训练精度: {np.mean(accuracies) * 100:.2f}')
                logger.info(f'验证精度: {avg_acc_val * 100:.2f} | 最佳验证精度: {best_avg_acc * 100:.2f} 在第 {best_epoch} 个周期')

    def load(self):
        """
        从磁盘加载模型参数
        """
        self.load_state_dict(torch.load(self.model_save_path))
