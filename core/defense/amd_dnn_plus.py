"""
@article{grosse2017statistical,
  title={On the (statistical) detection of adversarial examples},
  author={Grosse, Kathrin and Manoharan, Praveen and Papernot, Nicolas and Backes, Michael and McDaniel, Patrick},
  journal={arXiv preprint arXiv:1702.06280},
  year={2017}
}

@inproceedings{carlini2017adversarial,
  title={Adversarial examples are not easily detected: Bypassing ten detection methods},
  author={Carlini, Nicholas and Wagner, David},
  booktitle={Proceedings of the 10th ACM workshop on artificial intelligence and security},
  pages={3--14},
  year={2017}
}

This implementation is not an official version, but adapted from:
https://github.com/carlini/nn_breaking_detection
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

logger = logging.getLogger('core.defense.amd_dnn_plus')
logger.addHandler(ErrorHandler)

class AMalwareDetectionDNNPlus(nn.Module, DetectorTemplate):
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.95,
                 device='cpu', name='', **kwargs):
        # 调用父类构造函数
        nn.Module.__init__(self)
        DetectorTemplate.__init__(self)
        
        # 初始化输入参数
        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = ratio
        self.device = device
        self.name = name
        self.parse_args(**kwargs)

        # 恶意软件检测模型
        if md_nn_model is not None and isinstance(md_nn_model, nn.Module):
            # 如果提供了已经训练好的模型，就使用这个模型
            self.md_nn_model = md_nn_model
            self.is_fitting_md_model = False  # 默认情况下，模型已经被训练
        else:
            # 否则，创建一个新的恶意软件检测模型
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   self.n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            self.is_fitting_md_model = True  # 标记模型需要训练

        # 创建一个AMalwareDetectionDNNPlus模型
        self.amd_nn_plus = MalwareDetectionDNN(self.input_size,
                                               self.n_classes + 1,
                                               self.device,
                                               name,
                                               **kwargs)

        # 初始化一个不需要梯度的阈值参数
        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)

        # 模型保存路径
        self.model_save_path = path.join(config.get('experiments', 'amd_dnn_plus') + '_' + self.name,
                                         'model.pth')
        self.md_nn_model.model_save_path = self.model_save_path

        # 输出模型结构信息
        logger.info('========================================NN_PLUS model architecture==============================')
        logger.info(self)
        logger.info('===============================================end==============================================')


    def parse_args(self,
                   dense_hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
                   **kwargs
                   ):
        """
        解析传递给模型的参数。

        Parameters
        -------
        dense_hidden_units: List[int]，神经网络中全连接层的隐藏单元数量。
        dropout: float，dropout层的丢弃概率。
        alpha_: float，其他可能的超参数。
        **kwargs: dict，其他未明确定义的参数。
        """

        # 设置隐藏单元的数量
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
        else:
            raise TypeError("Expect a list of hidden units.")

        # 设置其他超参数
        self.dropout = dropout
        self.alpha_ = alpha_
        
        # 提取proc_number参数
        self.proc_number = kwargs['proc_number']
        
        # 如果还有其他未知参数，给出警告
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        模型的前向传播。

        Parameters
        -------
        x: torch.Tensor，输入数据。

        Returns
        -------
        logits: torch.Tensor，模型输出的原始分数。
        probabilities: torch.Tensor，模型输出的最后一个分类的概率。
        """

        # 通过模型获取logits
        logits = self.amd_nn_plus(x)
        
        # 从logits中减去其最大值，增加数值稳定性
        logits -= torch.amax(logits, dim=-1, keepdim=True).detach()  # 增加稳定性，可能有助于提高模型性能
        
        # 返回logits和概率
        if logits.dim() == 1:
            return logits, torch.softmax(logits, dim=-1)[-1]
        else:
            return logits, torch.softmax(logits, dim=-1)[:, -1]

    def predict(self, test_data_producer, indicator_masking=True):
        """
        预测标签并对检测器和指示器进行评估。

        参数
        --------
        test_data_producer: torch.DataLoader，测试数据加载器。
        indicator_masking: bool，是否过滤出低密度的样本或遮蔽它们的值。
        """
        # 进行模型推断，获取类中心（y_cent）、样本概率（x_prob）和真实标签（y_true）
        y_cent, x_prob, y_true = self.inference(test_data_producer)

        # 将预测的类标签和真实的类标签转为numpy数组
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        # 获取指示器标志，用于后续的遮蔽或过滤
        indicator_flag = self.indicator(x_prob).cpu().numpy()

        # 定义一个内部函数来评估模型的性能
        def measurement(_y_true, _y_pred):
            # 导入所需的评估指标库
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            
            # 计算并打印准确率和平衡准确率
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info(f"The accuracy on the test dataset is {accuracy * 100:.5f}%")
            logger.info(f"The balanced accuracy on the test dataset is {b_accuracy * 100:.5f}%")

            # 检查是否所有类都存在于真实标签中
            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("class absent.")
                return

            # 计算混淆矩阵并从中获取各项指标
            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(_y_true, _y_pred, average='binary')
            
            # 打印其他可能需要的评估指标
            logger.info(f"False Negative Rate (FNR) is {fnr * 100:.5f}%, \
                        False Positive Rate (FPR) is {fpr * 100:.5f}%, F1 score is {f1 * 100:.5f}%")

        # 首次进行评估
        measurement(y_true, y_pred)

        # 根据indicator_masking决定如何处理指示器
        if indicator_masking:
            # 排除带有“不确定”响应的示例
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # 在这里不是过滤掉样本，而是将其预测重置为1
            y_pred[~indicator_flag] = 1.

        # 打印指示器状态和阈值信息
        logger.info('The indicator is turning on...')
        logger.info(f'The threshold is {self.tau.item():.5}')

        # 再次进行评估
        measurement(y_true, y_pred)

    def inference(self, test_data_producer):
        """
        对测试数据进行推断。
        
        @param test_data_producer: 一个torch.DataLoader，用于生产测试数据批次。
        @return: 返回预测的类别概率、某种概率度量（可能是样本的确定性或其他度量）以及真实标签。
        """
        y_cent, x_prob = [], []
        gt_labels = []  # 存储真实的标签
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度，因为我们只是在做推断
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)  # 将数据移动到指定设备上
                logits, x_cent = self.forward(x)  # 获得模型输出
                y_cent.append(F.softmax(logits, dim=-1)[:, :2])  # 获得前两类的softmax输出
                x_prob.append(x_cent)  # 存储某种概率度量或确定性度量
                gt_labels.append(y)  # 存储真实标签

        # 将结果聚合成一个tensor
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        return y_cent, x_prob, gt_labels

    def inference_batch_wise(self, x):
        """
        批量-wise推断方法。给定输入x，返回对应的推断结果。
        
        @param x: 输入数据的Tensor
        @return: 返回预测的类别概率和某种概率度量。
        """
        assert isinstance(x, torch.Tensor)
        self.eval()  # 设置模型为评估模式
        logits, g = self.forward(x)  # 获得模型输出
        softmax_output = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        if len(softmax_output.shape) == 1:
            x_cent = softmax_output[:2]  # 如果是一维数组，只取前两个值
        else:
            x_cent = softmax_output[:, :2]  # 如果是二维数组，则取前两列

        return x_cent, g.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        """
        返回tau的值，tau可能是一个决策阈值或其他参数。
        
        @param y_pred: 可选的，预测的标签。在这里它似乎没有被使用。
        @return: 返回tau的值。
        """
        return self.tau


    # indicator 函数是基于概率值 x_prob 和阈值 tau 来判断给定的样本是否被认为是原始的。
    # 这可能在检测对抗性样本时很有用，因为对抗性样本可能有与原始样本不同的概率特征。
    def indicator(self, x_prob, y_pred=None):
        """
        根据样本的概率值x_prob和阈值tau，判断样本是否为原始样本。
        
        :@param x_prob: 样本的概率值。
        :@param y_pred: 可选的，预测的标签，但在此函数中未使用。
        :@return: 如果样本是原始样本，返回True，否则返回False。
        """
        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob <= self.tau
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # get_threshold 函数的目的是从验证数据集中计算并设置阈值 tau，
    # 这个阈值之后将被用于 indicator 函数来区分原始和对抗性样本。
    # 这里，阈值是基于给定比率的概率分位数，
    # 例如，如果 ratio 是0.95，那么 tau 将被设置为所有验证样本概率值中的第95个百分位数。
    def get_threshold(self, validation_data_producer, ratio=None):
        """
        获取对抗性检测的阈值。
        
        :@param validation_data_producer: 对象，用于生成验证数据集的迭代器。
        :@param ratio: 比率，用于确定要设置为阈值的概率分位数。
        """
        # 设置模型为评估模式
        self.eval()  
        
        # 如果没有指定比率，则使用类的默认比率
        ratio = ratio if ratio is not None else self.ratio  
        
         # 确保比率在[0, 1]范围内
        assert 0 <= ratio <= 1
        
        # 用于存储所有验证样本的概率值的列表 
        probabilities = [] 
        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                # 转化为torch tensor并移至指定设备
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.device) 
                
                # 获取模型的输出，_1表示我们不关心这个输出，只关心x_cent
                _1, x_cent = self.forward(x_val)
                
                # 添加到概率列表 
                probabilities.append(x_cent)  
                
            # 将所有概率连接并排序
            s, _ = torch.sort(torch.cat(probabilities, dim=0)) 
            
            # 获取要设置为阈值的概率的索引
            i = int((s.shape[0] - 1) * ratio)  
            assert i >= 0
            
            # 设置tau的值
            self.tau[0] = s[i]  


    def fit(self, train_data_producer, validation_data_producer, attack, attack_param,
            epochs=50, lr=0.005, weight_decay=0., verbose=True):
        """
        Train the alarm, pick the best model according to the validation results

        Parameters
        ----------
        @param train_data_producer: Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param attack, attack model, expect Max or Stepwise_Max
        @param attack_param, parameters used by the attack model
        @param epochs, Integer, epochs
        @param lr, Float, learning rate for Adam optimizer
        @param weight_decay, Float, penalty factor
        @param verbose: Boolean, whether to show verbose logs
        """
        # 训练恶意软件检测器
        if self.is_fitting_md_model:
            # 对恶意软件检测神经网络模型进行训练
            self.md_nn_model.fit(train_data_producer,
                                 validation_data_producer, epochs, lr, weight_decay)

        # 检查是否提供了攻击策略
        if attack is not None:
            # 确保攻击方法是Max或StepwiseMax之一
            assert isinstance(attack, (Max, StepwiseMax))
            # 如果攻击实例有‘is_attacker’属性，确保其值为False
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker

        # 使用Adam优化器初始化优化过程
        optimizer = optim.Adam(self.amd_nn_plus.parameters(), lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.  # 用于存储最佳的平均准确率
        best_epoch = 0  # 用于存储最佳准确率对应的轮次
        total_time = 0.  # 用于存储总训练时间
        pertb_train_data_list = []  # 用于存储训练数据的对抗样本
        pertb_val_data_list = []  # 用于存储验证数据的对抗样本
        nbatches = len(train_data_producer)  # 获取训练数据批次的数量
        logger.info("Training model with extra class ...")  # 训练时日志记录
        self.md_nn_model.eval()  # 将恶意软件检测模型设置为评估模式（不使用dropout等）

        # 开始训练过程
        for i in range(epochs):
            self.amd_nn_plus.train()  # 将模型设置为训练模式
            losses, accuracies = [], []  # 用于存储每个批次的损失和准确率

            # 遍历所有的训练数据批次
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # 将数据移动到指定的设备上（例如GPU）
                x_train, y_train = utils.to_device(x_train.double(), y_train.long(), self.device)

                # 开始生成异常数据
                start_time = time.time()
                # 如果当前批次的索引超过了对抗样本列表的长度
                if idx_batch >= len(pertb_train_data_list):
                    # 使用给定的攻击策略生成对抗样本
                    pertb_x = attack.perturb(self.md_nn_model, x_train, y_train,
                                             **attack_param
                                             )
                    # 对对抗样本进行四舍五入处理（针对StepwiseMax方法生成的连续样本）
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)
                    # 检查生成的对抗样本与原始样本之间是否有任何差异
                    trivial_atta_flag = torch.sum(torch.abs(x_train - pertb_x), dim=-1)[:] == 0.
                    # 如果所有对抗样本都与原始样本相同，则跳过这个批次
                    if torch.all(trivial_atta_flag):
                        pertb_train_data_list.append([])
                        continue
                    # 只保存那些与原始样本不同的对抗样本
                    pertb_x = pertb_x[~trivial_atta_flag]
                    # 将对抗样本添加到列表中
                    pertb_train_data_list.append(pertb_x.detach().cpu().numpy())
                else:
                    # 使用之前存储的对抗样本
                    pertb_x = pertb_train_data_list[idx_batch]
                    if len(pertb_x) == 0:
                        continue
                    pertb_x = torch.from_numpy(pertb_x).to(self.device)

                # 将原始数据和对抗样本合并
                x_train = torch.cat([x_train, pertb_x], dim=0)
                # 计算扩展后的批次大小
                batch_size_ext = x_train.shape[0]
                # 为对抗样本创建新的标签（这里为2）
                y_train = torch.cat([y_train, 2 * torch.ones((pertb_x.shape[0],), dtype=torch.long, device=self.device)])
                # 随机打乱数据和标签的顺序
                idx = torch.randperm(batch_size_ext)
                x_train = x_train[idx]
                y_train = y_train[idx]

                # 清除之前的梯度
                optimizer.zero_grad()
                # 获取模型的预测输出
                logits, _1 = self.forward(x_train)
                # 计算交叉熵损失
                loss_train = F.cross_entropy(logits, y_train)
                # 反向传播
                loss_train.backward()
                # 更新权重
                optimizer.step()
                # 更新总训练时间
                total_time = total_time + time.time() - start_time
                # 计算训练准确率
                acc_train = (logits.argmax(1) == y_train).sum().item()
                acc_train = acc_train / (len(x_train))
                # 将当前批次的损失和准确率添加到列表中
                losses.append(loss_train.item())
                accuracies.append(acc_train)

                # 如果设置了详细输出，记录训练进度和结果
                if verbose:
                    mins, secs = int(total_time / 60), int(total_time % 60)
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}%.')

            # 将模型设置为评估模式，用于验证
            self.amd_nn_plus.eval()

            # 初始化一个列表，用于存储每个批次的验证准确率
            avg_acc_val = []

            # 遍历所有的验证数据批次
            for idx, (x_val, y_val) in enumerate(validation_data_producer):
                # 将数据移动到指定的设备上（例如GPU）
                x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)

                # 为验证数据生成对抗样本 ⭐
                if idx >= len(pertb_val_data_list):
                    pertb_x = attack.perturb(self.md_nn_model, x_val, y_val,
                                            **attack_param
                                            )
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)
                    # 检查生成的对抗样本与原始样本之间是否有任何差异
                    trivial_atta_flag = torch.sum(torch.abs(x_val - pertb_x), dim=-1)[:] == 0.
                    # 断言至少有一个对抗样本与原始样本不同
                    assert (not torch.all(trivial_atta_flag)), 'No modifications.'
                    # 只保存那些与原始样本不同的对抗样本
                    pertb_x = pertb_x[~trivial_atta_flag]
                    pertb_val_data_list.append(pertb_x.detach().cpu().numpy())
                else:
                    # 使用之前存储的对抗样本
                    pertb_x = torch.from_numpy(pertb_val_data_list[idx]).to(self.device)

                # 将原始数据和对抗样本合并
                x_val = torch.cat([x_val, pertb_x], dim=0)
                # 为对抗样本创建新的标签（这里为2）
                y_val = torch.cat([y_val, 2 * torch.ones((pertb_x.shape[0],), device=self.device)])

                # 获取模型的预测输出
                logits, _1 = self.forward(x_val)
                # 计算验证准确率
                acc_val = (logits.argmax(1) == y_val).sum().item()
                acc_val = acc_val / (len(x_val))
                avg_acc_val.append(acc_val)

            # 计算整体验证准确率的平均值
            avg_acc_val = np.mean(avg_acc_val)

            # 检查当前的验证准确率是否比之前的最佳验证准确率要好
            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                # 计算并设置模型的决策阈值
                self.get_threshold(validation_data_producer)
                # 保存当前模型到磁盘
                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            # 输出当前轮次的训练损失、训练准确率和验证准确率
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        # 加载模型
        # 如果模型保存的路径不存在，那么抛出断言错误提示先进行训练
        assert path.exists(self.model_save_path), 'train model first'
        # 使用torch.load从指定路径加载模型的权重，并将其应用到当前模型上
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        # 如果模型保存的路径不存在
        if not path.exists(self.model_save_path):
            # 创建所需的目录结构
            utils.mkdir(path.dirname(self.model_save_path))
        # 将模型的状态字典保存到指定路径
        torch.save(self.state_dict(), self.model_save_path)
