from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
import os.path as path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import IntegratedGradients

import numpy as np

from core.defense.md_dnn import MalwareDetectionDNN
from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_input_convex_nn')
logger.addHandler(ErrorHandler)

# 定义进阶的恶意软件检测器，使用输入凸神经网络（Input Convex Neural Network, ICNN）
# 这段代码定义了一个进阶的恶意软件检测器类，该类使用输入凸神经网络（ICNN）结构。
# 其主要目标是为了提供一个能够检测恶意软件的高级模型，并对其内部的神经网络结构进行相应的优化与调整。

class AdvMalwareDetectorICNN(nn.Module, DetectorTemplate):
    # 初始化函数
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.98,
                 device='cpu', name='', **kwargs):
        # 调用父类的初始化函数
        nn.Module.__init__(self)
        DetectorTemplate.__init__(self)
        
        # 设置输入大小、类别数、比例、设备和名称等属性
        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = 0.98
        # print("self.ratio:", self.ratio)
        self.device = device
        self.name = name
        self.parse_args(**kwargs)
        
        # 检查md_nn_model是否是nn.Module的实例
        if isinstance(md_nn_model, nn.Module):
            self.md_nn_model = md_nn_model
        else:
            kwargs['smooth'] = True
            # 如果不是，构建一个默认的DNN恶意软件检测模型
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            # 警告使用者：使用了自定义的基于NN的恶意软件检测器
            warnings.warn("Use a self-defined NN-based malware detector")
        
        # 检查模型是否有'smooth'属性
        if hasattr(self.md_nn_model, 'smooth'):
            # 如果模型不是平滑的，将ReLU替换为SELU
            if not self.md_nn_model.smooth:
                for name, child in self.md_nn_model.named_children():
                    if isinstance(child, nn.ReLU):
                        self.md_nn_model._modules['relu'] = nn.SELU()
        else:
            # 没有'smooth'属性的情况下，将ReLU替换为SELU
            for name, child in self.md_nn_model.named_children():
                if isinstance(child, nn.ReLU):
                    self.md_nn_model._modules['relu'] = nn.SELU()
        
        # 将模型移动到指定的设备上
        self.md_nn_model = self.md_nn_model.to(self.device)

        # 输入凸神经网络
        self.non_neg_dense_layers = []
        
        # 至少需要一个隐藏层
        if len(self.dense_hidden_units) < 1:
            raise ValueError("Expect at least one hidden layer.")
        
        # 创建非负的密集层
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[i],
                                                       self.dense_hidden_units[i + 1],
                                                       bias=False))
        self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[-1], 1, bias=False))
        
        # 注册非负的密集层
        for idx_i, dense_layer in enumerate(self.non_neg_dense_layers):
            self.add_module('non_neg_layer_{}'.format(idx_i), dense_layer)

        # 创建密集层
        self.dense_layers = []
        self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        for i in range(len(self.dense_hidden_units[1:])):
            self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[i]))
        self.dense_layers.append(nn.Linear(self.input_size, 1))
        
        # 注册密集层
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('layer_{}'.format(idx_i), dense_layer)

        # 创建参数tau并设置为不需要梯度
        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)

        # 设置模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'amd_icnn') + '_' + self.name,
                                         'model.pth')
        # 打印模型的结构信息
        logger.info('========================================icnn model architecture==============================')
        logger.info(self)
        logger.info('===============================================end==========================================')


    def parse_args(self,
                dense_hidden_units=None,  # 密集层隐藏单元的列表
                dropout=0.6,               # dropout率
                alpha_=0.2,                # alpha参数
                **kwargs                   # 其他关键字参数
                ):
        # 如果没有提供密集层的隐藏单元，则使用默认的[200, 200]
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
            
        # 如果提供的密集层隐藏单元是列表形式，则直接赋值
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
            
        # 如果提供的不是列表，则抛出类型错误
        else:
            raise TypeError("Expect a list of hidden units.")

        # 设置dropout率
        self.dropout = dropout
        # 设置alpha参数
        self.alpha_ = alpha_
        # 获取`proc_number`参数
        self.proc_number = kwargs['proc_number']
        # 如果提供了额外的关键字参数，且参数数量大于0，则记录警告信息
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    # 定义forward_f函数，该函数对输入x应用md_nn_model模型
    def forward_f(self, x):
        return self.md_nn_model(x)

    # 定义forward_g函数，该函数处理输入数据x并传递给密集层和非负密集层
    def forward_g(self, x):
        # 初始化prev_x为None，用于存储前一个x的值
        prev_x = None
        # 对每个密集层进行枚举
        for i, dense_layer in enumerate(self.dense_layers):
            # 初始化x_add列表，用于存储中间结果
            x_add = []
            
            # 将输入x通过当前的密集层
            x1 = dense_layer(x)
            
            # 将结果添加到x_add列表中
            x_add.append(x1)
            
            # 如果prev_x不为None，表示不是第一个密集层
            if prev_x is not None:
                # 将前一个x通过非负密集层
                x2 = self.non_neg_dense_layers[i - 1](prev_x)
                # 将结果添加到x_add列表中
                x_add.append(x2)
                
            # 将x_add列表中的所有元素求和
            prev_x = torch.sum(torch.stack(x_add, dim=0), dim=0)
            
            # 如果不是最后一个密集层，则应用SELU激活函数
            if i < len(self.dense_layers):
                prev_x = F.selu(prev_x)
                
        # 改变输出的形状并返回
        return prev_x.reshape(-1)


    def forward(self, x):
        return self.forward_f(x), self.forward_g(x)

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x同时传递给forward_f和forward_g函数
        return self.forward_f(x), self.forward_g(x)

    # 定义预测函数
    def predict(self, test_data_producer, indicator_masking=True):
        """
        预测标签并对检测器和指示器进行评估

        参数:
        --------
        @param test_data_producer: torch.DataLoader，用于产生测试数据
        @param indicator_masking: 是否过滤掉低密度的示例或遮罩其值
        """
        # 从测试数据生成器中进行推断，获取中心预测值、概率和真实标签
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        # 获取预测值的最大索引作为预测结果
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        # 计算指示器标志
        indicator_flag = self.indicator(x_prob).cpu().numpy()

        # 定义评价函数
        def measurement(_y_true, _y_pred):
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            # 计算并打印准确率
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info("测试数据集的准确率为 {:.5f}%".format(accuracy * 100))
            logger.info("测试数据集的平衡准确率为 {:.5f}%".format(b_accuracy * 100))
            # 检查某个类是否完全缺失
            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("某个类别缺失。")
                return

            # 计算混淆矩阵并获取TP, TN, FP, FN
            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            # 计算F1分数
            f1 = f1_score(_y_true, _y_pred, average='binary')
            logger.info("假阴性率(FNR)为 {:.5f}%, 假阳性率(FPR)为 {:.5f}%, F1分数为 {:.5f}%".format(fnr * 100, fpr * 100, f1 * 100))

        # 对真实标签和预测标签进行评估
        measurement(y_true, y_pred)

        rtn_value = (y_pred == 0) & indicator_flag

        if indicator_masking:
            # 排除带有“不确定”响应的样本
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # 这里不是过滤掉示例，而是将预测重置为1
            y_pred[~indicator_flag] = 1.
        logger.info('指示器已开启...')
        logger.info('阈值为 {:.5}'.format(self.tau.item()))
        # 再次评估
        measurement(y_true, y_pred)

        return rtn_value


    # 定义推断函数
    def inference(self, test_data_producer):
        # 初始化三个空列表：y_cent用于存放预测的类别中心值，x_prob用于存放预测的概率值，gt_labels用于存放真实标签。
        y_cent, x_prob = [], []
        gt_labels = []
        
        # 将模型设置为评估模式
        self.eval()
        
        # 使用torch.no_grad()来指示PyTorch在此上下文中不计算梯度，这在推断时是常见的做法，可以节省内存并加速计算。
        with torch.no_grad():
            # 遍历测试数据生成器中的每一批数据
            for x, y in test_data_producer:
                # 将数据转移到设备上，并确保x的数据类型为double，y的数据类型为long
                x, y = utils.to_device(x.double(), y.long(), self.device)
                
                # 通过前向传播得到logits_f和logits_g
                logits_f, logits_g = self.forward(x)
                
                # 使用softmax函数计算logits_f的概率分布，并将其添加到y_cent列表中
                y_cent.append(torch.softmax(logits_f, dim=-1))
                
                # 将logits_g添加到x_prob列表中
                x_prob.append(logits_g)
                
                # 将真实标签添加到gt_labels列表中
                gt_labels.append(y)
        
        # 使用torch.cat将三个列表中的所有Tensor沿第0维度拼接起来
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        
        # 返回三个Tensor：y_cent, x_prob, gt_labels
        return y_cent, x_prob, gt_labels

    # 这段代码的主要目的是计算模型输入的重要性或贡献。
    # 整合梯度是一种解释机器学习模型的方法，它提供了一种方式来理解每个输入特性对预测结果的贡献是如何的。
    # 在这里，这种方法被用于两个不同的模型输出：分类任务(forward_f)和另一个可能与密度估计或某种特定任务有关的输出(forward_g)。
    def get_important_attributes(self, test_data_producer, indicator_masking=False):
        """
        获取输入的重要属性，使用整合梯度法 (integrated gradients)。

        邻接矩阵将被忽略。
        """
        # 存储分类任务的属性重要性
        attributions_cls = []
        # 存储其他任务(可能是密度估计或某种任务)的属性重要性
        attributions_de = []

        # 定义一个包装函数，用于分类任务的整合梯度计算
        def _ig_wrapper_cls(_x):
            logits = self.forward_f(_x)  # 获取模型对于输入x的预测
            return F.softmax(logits, dim=-1)  # 对预测进行softmax操作以得到概率值

        # 初始化整合梯度方法，针对分类任务
        ig_cls = IntegratedGradients(_ig_wrapper_cls)

        # 定义一个包装函数，用于其他任务的整合梯度计算
        def _ig_wrapper_de(_x):
            return self.forward_g(_x)

        # 初始化整合梯度方法，针对其他任务
        ig_de = IntegratedGradients(_ig_wrapper_de)

        # 遍历测试数据
        for i, (x, y) in enumerate(test_data_producer):
            x, y = utils.to_tensor(x, y, self.device)  # 将输入和标签转为张量
            x.requires_grad = True  # 为输入x设置梯度属性，以便后续计算梯度
            base_lines = torch.zeros_like(x, dtype=torch.double, device=self.device)  # 设置基线为全零
            base_lines[:, -1] = 1  # 修改基线的最后一个值为1
            # 计算分类任务的属性重要性
            attribution_bs = ig_cls.attribute(x,
                                              baselines=base_lines,
                                              target=1)  # target=1意味着我们计算对类别1的属性重要性
            attributions_cls.append(attribution_bs.clone().detach().cpu().numpy())

            # 计算其他任务的属性重要性
            attribution_bs = ig_de.attribute(x,
                                             baselines=base_lines
                                             )
            attributions_de.append(attribution_bs.clone().detach().cpu().numpy())
        
        # 将所有批次的结果合并为一个数组
        return np.vstack(attributions_cls), np.vstack(attributions_de)
    
    def inference_batch_wise(self, x):
        """
        返回分类的概率和g模型的输出。
        """
        assert isinstance(x, torch.Tensor)  # 断言确保输入是torch.Tensor类型
        self.eval()  # 将模型设置为评估模式
        logits_f, logits_g = self.forward(x)  # 获取f和g模型的输出
        # 对f模型的输出进行softmax操作以获得分类概率，并将结果转移到CPU上
        return torch.softmax(logits_f, dim=-1).detach().cpu().numpy(), logits_g.detach().cpu().numpy()


    def get_tau_sample_wise(self, y_pred=None):
        return self.tau  # 返回tau，即决策阈值


    def indicator(self, x_prob, y_pred=None):
        """
        判断一个样本是否是原始的。
        """
        # print("self.tau:", self.tau)
        if isinstance(x_prob, np.ndarray):  # 判断输入是否为numpy数组
            x_prob = torch.tensor(x_prob, device=self.device)  # 转换numpy数组为torch.Tensor
            # 判断每个样本的概率是否小于或等于tau，并返回结果
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):  # 判断输入是否为torch.Tensor
            return x_prob <= self.tau  # 判断每个样本的概率是否小于或等于tau，并返回结果
        else:
            # 如果输入既不是numpy数组也不是torch.Tensor，抛出一个类型错误
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # 简而言之，该方法计算模型的输出概率，对这些概率进行排序，然后基于所提供的ratio来确定阈值。
    # 当模型的输出低于这个阈值时，模型将认为输入是对抗的。
    def get_threshold(self, validation_data_producer, ratio=None):
        """
        获取用于对抗检测的阈值。
        
        参数:
        --------
        validation_data_producer : Object
            用于生产验证数据集的迭代器。
        ratio : float, 可选
            用于计算阈值的比率，默认为self.ratio。

        """
        self.eval()  # 将模型设置为评估模式
        # 如果未提供ratio，则使用self.ratio作为默认值
        ratio = ratio if ratio is not None else self.ratio
        
        # 断言确保ratio的值在[0,1]范围内
        assert 0 <= ratio <= 1
        probabilities = []  # 用于存储模型输出的概率值
        with torch.no_grad():  # 在不计算梯度的情况下
            for x_val, y_val in validation_data_producer:  # 从验证数据生成器中获取数据
                # 将输入数据和标签转换为适当的数据类型，并移动到指定的设备上
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.device)
                # 获取g模型的输出
                x_logits = self.forward_g(x_val)
                # 将模型输出添加到概率列表中
                probabilities.append(x_logits)
            # 对所有模型输出进行排序
            s, _ = torch.sort(torch.cat(probabilities, dim=0))
            # 计算索引i，它基于所提供的比率确定了阈值在排序输出中的位置
            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0  # 确保i是一个有效的索引
            # 设置模型的阈值tau为s[i]，即比率确定的阈值
            self.tau[0] = s[i]


    def reset_threshold(self):
        """
        重置模型的阈值为0。
        """
        self.tau[0] = 0.

    # 这个自定义的损失函数旨在同时训练模型以准确地分类原始样本，并检测出对抗样本。这是通过结合两种损失来实现的，其中每种损失都有其权重。
    def customize_loss(self, logits_x, labels, logits_adv_x, labels_adv, beta_1=1, beta_2=1):
        """
        自定义的损失函数，结合分类损失和对抗损失。

        参数:
        --------
        logits_x : torch.Tensor
            原始样本的模型输出。
        labels : torch.Tensor
            原始样本的真实标签。
        logits_adv_x : torch.Tensor
            对抗样本的模型输出。
        labels_adv : torch.Tensor
            对抗样本的真实标签。
        beta_1 : float, 可选
            原始样本损失的权重，默认为1。
        beta_2 : float, 可选
            对抗样本损失的权重，默认为1。

        返回:
        --------
        torch.Tensor
            计算得到的总损失。

        """
        # 如果有对抗样本，计算对抗损失。否则，将其设置为0。
        if logits_adv_x is not None and len(logits_adv_x) > 0:
            G = F.binary_cross_entropy_with_logits(logits_adv_x, labels_adv)
        else:
            G = 0

        # 如果有原始样本，计算分类损失。否则，将其设置为0。
        if logits_x is not None and len(logits_x) > 0:
            F_ = F.cross_entropy(logits_x, labels)
        else:
            F_ = 0

        # 结合两种损失，使用beta_1和beta_2作为权重
        return beta_1 * F_ + beta_2 * G


    # 这段代码描述了训练过程。它首先在每个时期开始时对模型进行训练，然后对每一个批次的数据进行训练。
    # 这里的亮点是它还生成了带有椒盐噪声的数据，并对其进行了分类。
    # 最后，它计算了每个批次的损失和准确率，并可能将其记录在日志中。
    
    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., verbose=True):
        """
        训练恶意软件和对抗检测器，根据验证结果选择最佳模型。

        参数:
        --------
        train_data_producer: 对象
            用于生成训练批次数据的迭代器。
        validation_data_producer: 对象
            用于生成验证数据集的迭代器。
        epochs: 整数
            训练的迭代次数，默认为100。
        lr: 浮点数
            Adam优化器的学习率，默认为0.005。
        weight_decay: 浮点数
            惩罚因子，默认为0。
        verbose: 布尔值
            是否显示详细日志，默认为True。
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.  # 初始化最佳平均准确率
        best_epoch = 0
        total_time = 0.  # 累计训练时间
        nbatches = len(train_data_producer)
        
        # 开始训练
        for i in range(epochs):
            self.train()  # 将模型设为训练模式
            losses, accuracies = [], []

            # 迭代训练批次数据
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # 数据移动到指定设备
                x_train, y_train = utils.to_device(x_train.double(), y_train.long(), self.device)
                
                # 为g网络生成数据
                # 1. 添加椒盐噪声
                x_train_noises = torch.clamp(x_train + utils.psn(x_train, np.random.uniform(0, 0.5)), min=0., max=1.)
                x_train_ = torch.cat([x_train, x_train_noises], dim=0)
                y_train_ = torch.cat([torch.zeros(x_train.shape[:1]), torch.ones(x_train.shape[:1])]).double().to(self.device)
                idx = torch.randperm(y_train_.shape[0])
                x_train_ = x_train_[idx]
                y_train_ = y_train_[idx]

                # 开始一次训练迭代
                start_time = time.time()
                optimizer.zero_grad()
                logits_f = self.forward_f(x_train)
                logits_g = self.forward_g(x_train_)
                loss_train = self.customize_loss(logits_f, y_train, logits_g, y_train_)
                loss_train.backward()
                optimizer.step()
                
                # 约束条件
                constraint = utils.NonnegWeightConstraint()
                for name, module in self.named_modules():
                    if 'non_neg_layer' in name:
                        module.apply(constraint)
                
                total_time = total_time + time.time() - start_time
                
                # 计算准确率
                acc_f_train = (logits_f.argmax(1) == y_train).sum().item() / x_train.size()[0]
                acc_g_train = ((F.sigmoid(logits_g) >= 0.5) == y_train_).sum().item() / x_train_.size()[0]
                
                # 更新记录
                losses.append(loss_train.item())
                accuracies.append(acc_f_train)
                accuracies.append(acc_g_train)
                
                # 如果需要，打印详细日志
                if verbose:
                    mins, secs = int(total_time / 60), int(total_time % 60)
                    logger.info(f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}% & {acc_g_train * 100:.2f}%.')
            
            # 设置模型为评估模式
            self.eval()
            
            # 初始化一个列表用于保存每批验证数据的准确率
            avg_acc_val = []
            
            # 禁用梯度计算，以加速计算并减少内存使用
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    # 数据移到指定设备
                    x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)
                    
                    # 为g网络生成数据（带有椒盐噪声）
                    x_val_noises = torch.clamp(x_val + utils.psn(x_val, np.random.uniform(0, 0.5)), min=0., max=1.)
                    x_val_ = torch.cat([x_val, x_val_noises], dim=0)
                    y_val_ = torch.cat([torch.zeros(x_val.shape[:1]), torch.ones(x_val.shape[:1])]).long().to(self.device)
                    
                    # 获取预测的标签
                    logits_f = self.forward_f(x_val)
                    logits_g = self.forward_g(x_val_)
                    
                    # 计算f网络的准确率
                    acc_val = (logits_f.argmax(1) == y_val).sum().item() / x_val.size()[0]
                    avg_acc_val.append(acc_val)
                    
                    # 计算g网络的准确率
                    acc_val_g = ((F.sigmoid(logits_g) >= 0.5) == y_val_).sum().item() / x_val_.size()[0]
                    avg_acc_val.append(acc_val_g)
                
                # 计算平均准确率
                avg_acc_val = np.mean(avg_acc_val)

            # 如果当前模型的验证准确率是迄今为止的最佳，则保存该模型
            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                # 获取阈值
                self.get_threshold(validation_data_producer)
                # 保存模型
                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            # 如果需要，显示训练和验证的详细信息
            if verbose:
                logger.info(f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        # load model
        assert path.exists(self.model_save_path), 'train model first'
        # ckpt = torch.load(self.model_save_path)
        # self.tau = ckpt['tau']
        # self.md_nn_model.load_state_dict(ckpt['md_model'])
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        # torch.save({
        #     'tau': self.tau,
        #     'md_model': self.md_nn_model.state_dict(),
        #     'amd_model': self.state_dict()
        # }, self.model_save_path
        # )
        torch.save(self.state_dict(), self.model_save_path)

