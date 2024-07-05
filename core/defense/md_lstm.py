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
logger = logging.getLogger('core.defense.lstm')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)


class MalwareDetectionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, n_classes=2, seq_len=1, device='cpu', name='LSTM', **kwargs):
        """
        初始化恶意软件检测器

        参数:
        ----------
        @param input_dim: 整数，每个时间步的输入特征维度。
        @param hidden_dim: 整数，LSTM隐藏状态的维度。
        @param n_classes: 整数，表示分类的数量，例如二分类问题中n=2。
        @param seq_len: 整数，序列的长度。
        @param device: 字符串，可以是'cpu'或'cuda'，表示模型应该在CPU还是GPU上运行。
        @param name: 字符串，用于命名模型。
        """
        super(MalwareDetectionLSTM, self).__init__()  # 调用父类初始化
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.device = device
        self.name = name

        # print(**kwargs)
        self.parse_args(**kwargs)   # 解析额外参数

        # 定义LSTM层
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # 定义输出层
        self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)

        # 根据参数选择使用SELU或ReLU激活函数
        if self.smooth:
            self.activation_func = F.selu  # 使用SELU激活函数
        else:
            self.activation_func = F.relu  # 使用ReLU激活函数

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'md_lstm') + '_' + self.name,
                                         'model.pth')
        # 日志中打印模型的结构信息
        logger.info(
            '======================================lstm model architecture===============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self, dropout=0.6, alpha_=0.2, smooth=False, **kwargs):
        """
        解析并设置网络的超参数。
        """

        # 设置dropout, alpha和smooth参数
        self.dropout = dropout
        self.alpha_ = alpha_
        self.smooth = smooth

        # 从kwargs中获取并设置proc_number
        self.proc_number = kwargs.get('proc_number', None)  # 如果不存在，则返回None

        # 如果还有其他参数，记录警告，因为这些参数可能是未知的
        if len(kwargs) > 0:
            # 请注意，您需要定义logger或删除下面的行
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        使输入数据 x 通过神经网络
        """
        x = x.view(-1, self.seq_len, self.input_dim)
        lstm_out, _ = self.lstm(x)
        # 只取最后一个时间步的输出
        latent_representation = lstm_out[:, -1, :]
        # 对处理过的数据进行 dropout 操作，用于防止过拟合
        latent_representation = F.dropout(
            latent_representation, self.dropout, training=self.training)
        logits = self.output_layer(latent_representation)
        # print("logits.shape:", logits.shape)
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
                # 确保数据具有正确的形状 [batch_size, seq_len, input_dim]
                x = x.view(-1, self.seq_len, self.input_dim)
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
                # 确保数据具有正确的形状 [batch_size, seq_len, input_dim]
                x = x.view(-1, self.seq_len, self.input_dim)
                # 将数据转移到指定的设备（CPU或GPU）并调整数据类型
                x, y = utils.to_device(x.double(), y.long(), self.device)
                # 得到每一批数据的logits
                logits = self.forward(x)
                # 使用softmax函数得到每一批数据的置信度，并将其添加到confidences列表中
                confidences.append(F.softmax(logits, dim=-1))
                # 将每一批数据的真实标签添加到gt_labels列表中
                gt_labels.append(y)

        return confidences, gt_labels

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

        # 确保数据具有正确的形状 [batch_size, seq_len, input_dim]
        x = x.view(-1, self.seq_len, self.input_dim)

        # 获得模型的输出
        logit = self.forward(x)

        # 返回每个样本的置信度和一个与logit形状相同的全1数组（表示恶意软件样本）
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        训练恶意软件检测器的LSTM模型，根据验证集上的交叉熵损失选择最佳模型。
        """
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        nbatches = len(train_data_producer)

        for i in range(epochs):
            self.train()
            losses, accuracies = [], []

            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # print("x_train.shape", x_train.shape)
                # 修改这部分以适应LSTM的输入格式 [batch_size, seq_len, input_dim]
                x_train = x_train.view(-1, self.seq_len, self.input_dim)
                x_train, y_train = utils.to_device(
                    x_train.double(), y_train.long(), self.device)
                # print("x_train.shape", x_train.shape)

                start_time = time.time()
                optimizer.zero_grad()
                logits = self.forward(x_train)
                # print("logits.shape", logits.shape)
                loss_train = self.customize_loss(logits, y_train)
                loss_train.backward()
                optimizer.step()
                total_time += time.time() - start_time
                acc_train = (logits.argmax(1) == y_train).sum(
                ).item() / x_train.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    logger.info(
                        f'小批次： {i * nbatches + idx_batch + 1}/{epochs * nbatches} | 训练时间为 {mins:.0f} 分钟, {secs} 秒。')
                    logger.info(
                        f'训练损失（小批次级别）: {losses[-1]:.4f} | 训练精度: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []

            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    # 同样确保数据形状适应LSTM的输入
                    x_val = x_val.view(-1, self.seq_len, self.input_dim)
                    x_val, y_val = utils.to_device(
                        x_val.double(), y_val.long(), self.device)
                    logits = self.forward(x_val)
                    acc_val = (logits.argmax(1) == y_val).sum(
                    ).item() / x_val.size()[0]
                    avg_acc_val.append(acc_val)

                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))
                torch.save(self.state_dict(), self.model_save_path)
                if verbose:
                    print(f'模型保存在路径： {self.model_save_path}')

            if verbose:
                logger.info(
                    f'训练损失（周期级别）: {np.mean(losses):.4f} | 训练精度: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'验证精度: {avg_acc_val * 100:.2f} | 最佳验证精度: {best_avg_acc * 100:.2f} 在第 {best_epoch} 个周期')

    def load(self):
        """
        从磁盘加载模型参数
        """
        self.load_state_dict(torch.load(self.model_save_path))
