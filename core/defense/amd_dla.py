"""
@inproceedings{sperl2020dla,
  title={DLA: dense-layer-analysis for adversarial example detection},
  author={Sperl, Philip and Kao, Ching-Yu and Chen, Peng and Lei, Xiao and B{\"o}ttinger, Konstantin},
  booktitle={2020 IEEE European Symposium on Security and Privacy (EuroS\&P)},
  pages={198--215},
  year={2020},
  organization={IEEE}
}

This implementation is not an official version, but adapted from:
https://github.com/v-wangg/OrthogonalPGD/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from core.attack.max import Max
from core.attack.stepwise_max import StepwiseMax
from core.defense.md_dnn import MalwareDetectionDNN
from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_dla')
logger.addHandler(ErrorHandler)


class AMalwareDetectionDLA(nn.Module, DetectorTemplate):
    # 初始化函数
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.95,
                 device='cpu', name='', **kwargs):
        # 初始化PyTorch模型
        nn.Module.__init__(self)
        # 初始化检测器模板
        DetectorTemplate.__init__(self)

        # 设置输入大小、分类数、比例、设备和名称
        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = ratio
        self.device = device
        self.name = name
        # 解析传入的其他参数
        self.parse_args(**kwargs)

        # 恶意软件检测器
        # 如果md_nn_model已经提供，并且是nn.Module的实例，直接使用
        if md_nn_model is not None and isinstance(md_nn_model, nn.Module):
            self.md_nn_model = md_nn_model
            # 默认为模型已经训练好了
            self.is_fitting_md_model = False
        # 否则，使用MalwareDetectionDNN创建一个新的模型
        else:
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            # 设置为需要训练模型
            self.is_fitting_md_model = True

        # 至少需要一个隐藏层
        assert len(
            self.dense_hidden_units) >= 1, "Expected at least one hidden layer."

        # 初始化报警模型
        self.alarm_nn_model = TorchAlarm(
            input_size=sum(self.md_nn_model.dense_hidden_units))

        # 定义一个参数tau，初始为0
        self.tau = nn.Parameter(torch.zeros(
            [1, ], device=self.device), requires_grad=False)

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'amd_dla') + '_' + self.name,
                                         'model.pth')
        self.md_nn_model.model_save_path = self.model_save_path
        # 记录模型结构信息
        logger.info(
            '========================================DLA model architecture==============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    # 该方法用于解析和设置模型的参数

    def parse_args(self,
                   dense_hidden_units=None,  # 隐藏层单元数
                   dropout=0.6,  # Dropout参数
                   alpha_=0.2,  # alpha参数
                   **kwargs  # 其他参数
                   ):
        # 如果没有传入dense_hidden_units，则默认设置为[200, 200]
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]

        # 如果传入的dense_hidden_units是一个列表，直接使用
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units

        # 否则抛出TypeError
        else:
            raise TypeError("Expect a list of hidden units.")

        # 设置dropout和alpha值
        self.dropout = dropout
        self.alpha_ = alpha_

        # 设置处理器数量
        self.proc_number = kwargs['proc_number']

        # 如果还有其他额外参数，发出警告
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    # 定义模型的前向传播函数
    def forward(self, x):
        # 初始化一个空列表用于存放隐藏层的激活值
        hidden_activations = []

        # 遍历除最后一个外的所有dense层
        for dense_layer in self.md_nn_model.dense_layers[:-1]:
            # 获取dense层的输出，并应用激活函数
            x = self.md_nn_model.activation_func(dense_layer(x))
            # 将激活值存入列表
            hidden_activations.append(x)
        # 对最后一个dense层进行操作
        logits = self.md_nn_model.dense_layers[-1](x)
        # 将所有隐藏层的激活值进行连接
        hidden_activations = torch.cat(hidden_activations, dim=-1)
        # 使用报警模型进行预测
        x_prob = self.alarm_nn_model(hidden_activations).reshape(-1)
        # 返回logits和x_prob
        return logits, x_prob

    def predict(self, test_data_producer, indicator_masking=True):
        """
        预测标签并对检测器及指示器进行评估

        参数
        --------
        @param test_data_producer, torch.DataLoader: 用于产生测试数据的 DataLoader。
        @param indicator_masking, bool: 是否过滤低密度的示例或屏蔽其值。
        """
        # 进行推断并获取中心值、概率和真实值
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        # 使用指示器进行评估
        indicator_flag = self.indicator(x_prob).cpu().numpy()

        # 定义测量函数，用于计算并输出各种性能指标
        def measurement(_y_true, _y_pred):
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)

            # 输出测试数据集上的准确性
            logger.info("测试数据集的准确度为 {:.5f}%".format(accuracy * 100))
            logger.info("测试数据集的平衡准确度为 {:.5f}%".format(b_accuracy * 100))

            # 如果测试数据中缺少某个类，则输出警告
            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("某个类别缺失。")
                return

            # 计算并输出其他评估指标，如 FNR、FPR 和 F1 分数
            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(_y_true, _y_pred, average='binary')
            logger.info("我们可能需要的其他评估指标：")
            logger.info("假负率(FNR)为 {:.5f}%, 假正率(FPR)为 {:.5f}%, F1 分数为 {:.5f}%".format(
                fnr * 100, fpr * 100, f1 * 100))

        # 首先进行完整的性能评估
        measurement(y_true, y_pred)

        # 根据是否使用指示器屏蔽来决定如何处理预测
        if indicator_masking:
            # 排除“不确定”响应的示例
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # 而不是过滤示例，此处将预测重置为1
            y_pred[~indicator_flag] = 1.

        # 输出指示器相关的日志
        logger.info('指示器已开启...')
        logger.info('阈值为 {:.5}'.format(self.tau.item()))

        # 再次进行性能评估，但这次使用了指示器
        measurement(y_true, y_pred)

    def inference(self, test_data_producer):
        # 初始化空列表，用于存储每个batch的中心值、概率值和真实标签
        y_cent, x_prob = [], []
        gt_labels = []
        self.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 确保不计算梯度，这在评估和测试阶段是常见的
            for x, y in test_data_producer:  # 从数据加载器中获取数据
                # 将数据移至适当的设备，并确保其具有正确的数据类型
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits, x_logits = self.forward(x)  # 使用模型前向传播
                y_cent.append(F.softmax(logits, dim=-1))  # 使用softmax获取类别概率
                x_prob.append(x_logits)  # 将概率值存入列表
                gt_labels.append(y)  # 存储真实标签

        # 将所有batch的结果连接起来，返回为单个张量
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        return y_cent, x_prob, gt_labels  # 返回中心值、概率值和真实标签

    def inference_batch_wise(self, x):
        assert isinstance(x, torch.Tensor)  # 确保输入是张量
        self.eval()  # 将模型设置为评估模式
        logits, x_prob = self.forward(x)  # 使用模型前向传播
        # 返回类别概率和概率值，都转换为numpy数组
        return torch.softmax(logits, dim=-1).detach().cpu().numpy(), x_prob.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau  # 返回阈值

    def indicator(self, x_prob, y_pred=None):
        """
        返回 'True' 表示样本是原始的, 否则返回 'False'。
        """
        if isinstance(x_prob, np.ndarray):
            # 将numpy数组转换为torch张量
            x_prob = torch.tensor(x_prob, device=self.device)
            # 如果x_prob小于或等于预设定的阈值，则返回True，否则返回False
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob <= self.tau  # 如果x_prob是torch张量，则直接进行比较
        else:
            # 如果输入既不是numpy数组也不是torch张量，则抛出错误
            raise TypeError("Tensor or numpy.ndarray are expected.")

    def get_threshold(self, validation_data_producer, ratio=None):
        """
        获取用于对抗性检测的阈值
        :@param validation_data_producer: 对象，用于生成验证数据集的迭代器
        """
        self.eval()  # 将模型设置为评估模式
        ratio = ratio if ratio is not None else self.ratio  # 如果没有提供ratio，就使用类的ratio属性
        assert 0 <= ratio <= 1  # 确保ratio在0到1之间
        probabilities = []
        with torch.no_grad():  # 确保不计算梯度
            for x_val, y_val in validation_data_producer:  # 从数据加载器中获取数据
                x_val, y_val = utils.to_tensor(
                    x_val.double(), y_val.long(), self.device)  # 将数据转换为张量并移到适当的设备
                _1, x_prob = self.forward(x_val)  # 使用模型前向传播
                probabilities.append(x_prob)  # 将概率值存入列表
            # 对概率值进行排序
            s, _ = torch.sort(torch.cat(probabilities, dim=0))
            # 根据ratio计算索引
            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0  # 确保索引是有效的
            self.tau[0] = s[i]  # 将阈值设置为对应的概率值

    # 该fit方法的核心目的是对报警模型进行训练。
    # 在训练过程中，它也生成了一些攻击数据，这可能是为了增强模型的鲁棒性。
    # 每个小批量中的数据都会经过正常和攻击数据的处理，以确保模型能在这两种情境下都表现良好。

    def fit(self, train_data_producer, validation_data_producer, attack, attack_param,
            epochs=100, lr=0.005, weight_decay=0., verbose=True):
        """
        训练报警模型，并根据验证结果选择最佳模型

        参数
        ----------
        @param train_data_producer: 对象，用于生成一批训练数据的迭代器
        @param validation_data_producer: 对象，用于生成验证数据集的迭代器
        @param attack, 攻击模型，期望为 Max 或 Stepwise_Max
        @param attack_param, 由攻击模型使用的参数
        @param epochs, 整数, 训练轮数
        @param lr, 浮点数, Adam优化器的学习率
        @param weight_decay, 浮点数, 惩罚系数
        @param verbose: 布尔值, 是否显示详细日志
        """

        # 训练恶意软件检测器
        if self.is_fitting_md_model:
            self.md_nn_model.fit(
                train_data_producer, validation_data_producer, epochs, lr, weight_decay)

        # 检查并断言攻击模型类型
        if attack is not None:
            assert isinstance(attack, (Max, StepwiseMax))
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker

        logger.info("Training alarm ...")

        # 设置优化器
        optimizer = optim.Adam(
            self.alarm_nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        pertb_train_data_list = []
        pertb_val_data_list = []
        nbatches = len(train_data_producer)
        self.md_nn_model.eval()

        # 开始训练过程
        for i in range(epochs):
            self.alarm_nn_model.train()
            losses, accuracies = [], []

            # 从数据生成器中读取数据
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train, y_train = utils.to_device(
                    x_train.double(), y_train.long(), self.device)
                batch_size = x_train.shape[0]

                # 产生异常数据
                start_time = time.time()
                if idx_batch >= len(pertb_train_data_list):
                    pertb_x = attack.perturb(
                        self.md_nn_model, x_train, y_train, **attack_param)
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)
                    trivial_atta_flag = torch.sum(
                        torch.abs(x_train - pertb_x), dim=-1)[:] == 0.
                    assert (not torch.all(trivial_atta_flag)
                            ), 'No modifications.'
                    pertb_x = pertb_x[~trivial_atta_flag]
                    pertb_train_data_list.append(
                        pertb_x.detach().cpu().numpy())
                else:
                    pertb_x = torch.from_numpy(
                        pertb_train_data_list[idx_batch]).to(self.device)

                # 合并数据
                x_train = torch.cat([x_train, pertb_x], dim=0)
                batch_size_ext = x_train.shape[0]

                # 生成标签
                y_train = torch.zeros((batch_size_ext, ), device=self.device)
                y_train[batch_size:] = 1

                idx = torch.randperm(y_train.shape[0])
                x_train = x_train[idx]
                y_train = y_train[idx]

                # 前向传播和计算损失
                optimizer.zero_grad()
                _1, x_logits = self.forward(x_train)
                loss_train = F.binary_cross_entropy_with_logits(
                    x_logits, y_train)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time

                # 计算准确率
                acc_g_train = ((torch.sigmoid(x_logits) >= 0.5)
                               == y_train).sum().item()
                acc_g_train = acc_g_train / batch_size_ext
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_g_train)

                # 如果需要显示详细日志
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches}'
                        f'| training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_g_train * 100:.2f}%.')

            # 将模型设置为评估模式
            self.alarm_nn_model.eval()
            avg_acc_val = []

            # 通过验证数据生成器迭代验证数据
            for idx, (x_val, y_val) in enumerate(validation_data_producer):
                # 将数据移到设备上（如GPU）
                x_val, y_val = utils.to_device(
                    x_val.double(), y_val.long(), self.device)
                batch_size_val = x_val.shape[0]

                # 为验证数据生成攻击样本
                if idx >= len(pertb_val_data_list):
                    pertb_x = attack.perturb(
                        self.md_nn_model, x_val, y_val, **attack_param)
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)

                    # 检查生成的攻击数据是否有真正的修改（即与原始数据是否有差异）
                    trivial_atta_flag = torch.sum(
                        torch.abs(x_val - pertb_x), dim=-1)[:] == 0.
                    assert (not torch.all(trivial_atta_flag)
                            ), 'No modifications.'
                    pertb_x = pertb_x[~trivial_atta_flag]

                    # 将攻击数据存储在列表中，以备后用
                    pertb_val_data_list.append(pertb_x.detach().cpu().numpy())
                else:
                    # 如果已经为这批验证数据生成了攻击样本，就直接从列表中取出
                    pertb_x = torch.from_numpy(
                        pertb_val_data_list[idx]).to(self.device)

                # 合并原始验证数据和攻击数据
                x_val = torch.cat([x_val, pertb_x], dim=0)
                batch_size_val_ext = x_val.shape[0]

                # 生成对应的标签，其中，攻击数据的标签为1
                y_val = torch.zeros((batch_size_val_ext,), device=self.device)
                y_val[batch_size_val:] = 1

                # 通过模型进行前向传播
                _1, x_logits = self.forward(x_val)

                # 计算模型的准确率
                acc_val = ((torch.sigmoid(x_logits) >= 0.5)
                           == y_val).sum().item()
                acc_val = acc_val / (2 * batch_size_val)
                avg_acc_val.append(acc_val)

            # 计算平均准确率
            avg_acc_val = np.mean(avg_acc_val)

            # 如果当前的平均准确率超过了之前的最佳值
            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                self.get_threshold(validation_data_producer)

                # 保存模型
                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            # 如果设置为显示详细日志，输出训练和验证的统计信息
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        # load model
        assert path.exists(self.model_save_path), 'train model first'
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        torch.save(self.state_dict(), self.model_save_path)


# 定义TorchAlarm类，该类继承自torch.nn.Module
class TorchAlarm(torch.nn.Module):
    # 初始化函数
    def __init__(self, input_size):
        # 调用父类的初始化方法
        super().__init__()

        # 定义一个神经网络层列表
        # 包含多个线性层与ReLU激活函数
        self.layers = torch.nn.ModuleList([
            # 第一个线性层，输入维度为input_size, 输出维度为112
            torch.nn.Linear(input_size, 112),
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(112, 100),         # 第二个线性层，输入维度为112, 输出维度为100
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(100, 300),         # 第三个线性层，输入维度为100, 输出维度为300
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(300, 200),         # 第四个线性层，输入维度为300, 输出维度为200
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(200, 77),          # 第五个线性层，输入维度为200, 输出维度为77
            torch.nn.ReLU(),                   # ReLU激活函数
            torch.nn.Linear(77, 1),            # 最后一个线性层，输入维度为77, 输出维度为1
        ])

    # 当这个类的实例被当作函数调用时，会执行此方法
    def __call__(self, x, training=False):
        # 如果x不是一个torch.Tensor，那么将x转化为torch.Tensor类型
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # 遍历layers中的每一层，将x传入，得到输出
        for layer in self.layers:
            x = layer(x)

        # 返回最后的输出
        return x
