# 使用未来版本特性，确保代码在Python2和Python3中有一致的行为
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入基础库
import random
import os.path as path

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data
from core.defense.amd_template import DetectorTemplate

# 导入Captum库，该库提供模型解释性工具，这里特别使用了集成梯度方法
from captum.attr import IntegratedGradients

# 导入NumPy
import numpy as np

# 从config模块中导入配置、日志和错误处理相关功能
from config import config, logging, ErrorHandler

# 导入自定义的工具模块
from tools import utils

from core.defense import Dataset

# 初始化日志记录器并设置其名称
logger = logging.getLogger('core.defense.fd_vae')

# 向日志记录器添加一个错误处理器，确保错误信息被适当捕获和处理
logger.addHandler(ErrorHandler)


def get_label(length, IsBenign):
    '''
        获取标签。
        length: 标签的数量。
        IsBen: 所需的标签是否为良性。
    '''
    # 确保长度值为整数
    assert length-int(length) < 0.01
    length = int(length)

    # 生成长度为length的全1和全0向量
    ones = np.ones((length, 1))
    zeros = np.zeros((length, 1))

    # 根据IsBenign的值，返回对应的标签
    if IsBenign:
        return np.column_stack((ones, zeros))  # 如果IsBenign为True，返回良性标签
    else:
        return np.column_stack((zeros, ones))  # 如果IsBenign为False，返回恶性标签


def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Variable: {name}, requires_grad: {param.requires_grad}")


class mu_sigma_MLP(nn.Module):
    # 初始化函数
    def __init__(self,
                 num_epoch=30,
                 learn_rate=1e-3,
                 z_dim=20,
                 name='mlp'
                 ):
        super(mu_sigma_MLP, self).__init__()  # 调用父类(nn.Module)的初始化函数

        self.num_epoch = num_epoch      # 设置训练的轮数
        self.batch_size = 128           # 设置每批数据的大小
        self.learn_rate = learn_rate    # 设置学习率
        self.z_dim = z_dim              # 设置输入数据的维度
        self.mu_wgt = 0.5               # 设置mu的权重
        self.sgm_wgt = 0.5              # 设置sigma的权重
        self.name = name                # 模型名称

        # 定义神经网络结构:
        # mu网络用于处理mu输入
        self.mu_net = self._build_net(n_hidden=1000)

        # sigma网络用于处理sigma输入
        self.sigma_net = self._build_net(n_hidden=1000)

        # 打印模型的结构
        print('================================mu_sigma_MLP model architecture==============================')
        print(self)
        print('===============================================end==========================================')

        # self.model_save_path = path.join(config.get('experiments', 'mlp') + '_' + "20231009-170626", 'model.pth')
        self.model_save_path = path.join(config.get(
            'experiments', 'mlp') + '_' + self.name, 'model.pth')

    # 定义神经网络的子结构，这个网络包括4个线性层和中间的激活函数

    def _build_net(self, n_hidden=128):
        # 使用Sequential构造一个串联的神经网络模块
        return nn.Sequential(
            nn.Linear(self.z_dim, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, 2)
        )

    # 前向传播函数
    def forward(self, mu_in, sigma_in):
        # 通过mu网络计算mu的输出
        y1 = self.mu_net(mu_in)

        # 通过sigma网络计算sigma的输出
        y2 = self.sigma_net(sigma_in)

        # 结合mu和sigma的输出，使用权重进行加权平均
        y = self.mu_wgt * y1 + self.sgm_wgt * y2

        return y

    def train_model(self, vae_model, train_data_producer, batch_size=128, n_epochs=50, verbose=True, device='cuda'):
        optimizer = optim.Adam(self.parameters())

        criterion_classification = nn.BCEWithLogitsLoss()

        # Reconstruction loss for VAE
        criterion_reconstruction = nn.MSELoss()
        best_accuracy = .0
        for epoch in range(n_epochs):
            for idx, (inputs, labels) in enumerate(train_data_producer):
                inputs, labels = inputs.to(device), labels.to(device)

                # Ensure inputs require gradients
                inputs.requires_grad_(True)

                optimizer.zero_grad()

                # Forward propagation
                y, muvae, sigmavae = vae_model.f(inputs)
                # print("muvae, sigmavae:", (muvae, sigmavae))
                outputs = self(muvae, sigmavae)

                loss_reconstruction = criterion_reconstruction(y, inputs)

                # Convert labels to one-hot encoding
                labels = labels.long()
                one_hot_labels = torch.zeros(
                    labels.size(0), outputs.size(1)).to(device)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

                loss_classification = criterion_classification(
                    outputs, one_hot_labels)

                # Total loss
                loss = loss_reconstruction + loss_classification

                # Check if loss requires gradient
                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss tensor does not require gradients.")

                loss.backward()
                optimizer.step()

                if idx % 500 == 0:
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    print(predicted)
                    correct = (predicted == labels).sum().item()
                    accuracy = correct / len(labels)

                    # Printing
                    logger.info(
                        f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy

                        # 检查模型保存路径是否存在，如果不存在，则创建
                        if not path.exists(self.model_save_path):
                            utils.mkdir(path.dirname(self.model_save_path))

                        # 保存当前的模型参数
                        torch.save(self.state_dict(), self.model_save_path)

                        # 如果开启了详细输出模式，显示模型保存路径
                        if verbose:
                            print(f'模型保存在路径： {self.model_save_path}')

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))


class VAE_2(nn.Module):
    def __init__(self,
                 dim_img=10000,
                 n_hidden=200,
                 dim_z=20,
                 KLW=5,
                 NLOSSW=10,
                 loss_type='1',
                 learn_rate=1e-3,
                 name='VAE_2'):
        super(VAE_2, self).__init__()

        # 初始化变量
        self.dim_img = dim_img          # 图片的维度
        self.n_hidden = n_hidden        # 隐藏层的神经元数量
        self.dim_z = dim_z              # 潜在空间的维度
        self.KLW = KLW                  # KL散度的权重
        self.NLOSSW = NLOSSW            # 新的损失函数的权重
        self.loss_type = loss_type      # 损失函数的类型
        self.learn_rate = learn_rate    # 学习率
        self.name = name                # 模型名称
        self.mu = -1
        self.sigma = -1

        # 高斯MLP编码器网络结构
        self.fc1 = nn.Linear(self.dim_img, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc2_mu = nn.Linear(n_hidden, self.dim_z)
        self.fc2_sigma = nn.Linear(n_hidden, self.dim_z)

        # 伯努利MLP解码器网络结构
        self.fc3 = nn.Linear(self.dim_z, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, self.dim_img)

        # 打印模型的结构
        print('========================================VAE model architecture==============================')
        print(self)
        print('===============================================end==========================================')

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get(
            'experiments', 'vae') + '_' + self.name, 'model.pth')
        # self.model_save_path = path.join(config.get('experiments', 'vae') + '_' + "20231008-184510", 'model.pth')

    def gaussian_MLP_encoder(self, x, keep_prob):
        h0 = F.elu(self.fc1(x))
        h0 = F.dropout(h0, p=1-keep_prob)

        h1 = F.tanh(self.fc2(h0))
        h1 = F.dropout(h1, p=1-keep_prob)

        # 计算均值mu和方差sigma
        mu = self.fc2_mu(h1)
        sigma = 1e-6 + F.softplus(self.fc2_sigma(h1))

        return mu, sigma

    def bernoulli_MLP_decoder(self, z, keep_prob):
        h0 = F.tanh(self.fc3(z))
        h0 = F.dropout(h0, p=1-keep_prob)

        h1 = F.elu(self.fc4(h0))
        h1 = F.dropout(h1, p=1-keep_prob)

        # 输出层使用Sigmoid激活函数，得到在[0,1]范围内的输出y
        y = torch.sigmoid(self.fc5(h1))
        return y

    def forward(self, x_hat, x, x_comp, label_x, label_x_comp, keep_prob):
        # 编码器部分:
        mu, sigma = self.gaussian_MLP_encoder(x_hat, keep_prob)
        # print("Shape of x_comp:", x_comp.shape)
        mu1, sigma1 = self.gaussian_MLP_encoder(x_comp, keep_prob)
        muvae, sigmavae = self.gaussian_MLP_encoder(x, keep_prob)

        # 更新 mu 和 sigma 的值
        self.mu = muvae
        self.sigma = sigmavae

        # 对z进行采样:
        # 利用高斯分布的重参数技巧采样z
        z = muvae + sigmavae * torch.randn_like(muvae)

        # 解码器部分:
        y = self.bernoulli_MLP_decoder(z, keep_prob)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        # 计算损失函数:
        marginal_likelihood = torch.sum(
            x * torch.log(y) + (1 - x) * torch.log(1 - y), 1)
        KL_divergence = 0.5 * \
            torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2) - 1, 1)
        vector_loss = torch.mean((label_x_comp - label_x)**2, 1)
        loss_bac = 60 * vector_loss
        loss_mean = torch.mean((mu - mu1)**2, 1)

        # 根据向量损失和均值损失计算总损失
        loss_0 = torch.mean(loss_mean * (1 - vector_loss))
        loss_1 = torch.mean(
            torch.abs(F.relu(loss_bac - loss_mean)) * vector_loss)

        # 计算ELBO (证据下界)
        ELBO = self.KLW * \
            torch.mean(marginal_likelihood) - torch.mean(KL_divergence)
        loss = -ELBO

        # 根据新的损失更新原损失
        new_loss = (loss_1 + loss_0) * self.NLOSSW
        if self.loss_type[0] == '1':
            loss = loss + new_loss

        return y, z, loss, torch.mean(marginal_likelihood), torch.mean(KL_divergence)

    def f(self, x):
        muvae, sigmavae = self.gaussian_MLP_encoder(x, 0.9)

        # 更新 mu 和 sigma 的值
        self.mu = muvae
        self.sigma = sigmavae

        # 对z进行采样:
        # 利用高斯分布的重参数技巧采样z
        z = muvae + sigmavae * torch.randn_like(muvae)

        # 解码器部分:
        y = self.bernoulli_MLP_decoder(z, 0.9)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        return y, muvae, sigmavae

    def train_model(self, train_data_producer, batch_size=128, n_epochs=10, verbose=True, device='cuda'):
        # 初始化优化器，这里使用Adam优化器
        optimizer = optim.Adam(self.parameters())

        ben_data_list = []
        mal_data_list = []

        # Traverse the DataLoader
        for _, (x_train, y_train) in enumerate(train_data_producer):
            ben_data = x_train[y_train == 0].to(device)
            mal_data = x_train[y_train == 1].to(device)

            ben_data_list.append(ben_data)
            mal_data_list.append(mal_data)
            mal_data_list.append(mal_data)  # 叠加两次

        # Concatenate all batches together
        ben_data_combined = torch.cat(ben_data_list, 0).to(device)
        mal_data_combined = torch.cat(mal_data_list, 0).to(device)

        # 获取较少的样本数，为了确保良性和恶意样本数量相同
        n_samples = min(len(ben_data_combined), len(mal_data_combined))
        total_batch = n_samples // batch_size

        # 开始训练循环
        for epoch in range(n_epochs):
            # 设置随机数种子
            random.seed(epoch)

            # 对数据进行随机排列
            indices_ben = torch.randperm(len(ben_data_combined))[:n_samples]
            indices_mal = torch.randperm(len(mal_data_combined))[:n_samples]

            # 获取随机排列后的数据
            ben_data_combined = ben_data_combined[indices_ben]
            mal_data_combined = mal_data_combined[indices_mal]

            # 获取批次的标签
            lbBen = get_label(batch_size, True)
            lbMal = get_label(batch_size, False)
            batch_label = np.row_stack((lbBen, lbMal))

            for i in range(total_batch):
                offset = (i * batch_size) % n_samples

                # 获取良性和恶意的批次数据
                batch_ben_input_s = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_mal_input_s = mal_data_combined[offset:(
                    offset + batch_size), :]
                batch_ben_input_s_cpu = batch_ben_input_s.cpu().numpy()
                batch_mal_input_s_cpu = batch_mal_input_s.cpu().numpy()
                batch_input = np.row_stack(
                    (batch_ben_input_s_cpu, batch_mal_input_s_cpu))

                # 合并输入数据和标签
                batch_input_wl = np.column_stack((batch_input, batch_label))
                np.random.shuffle(batch_input_wl)  # 打乱合并后的数据

                # 分离输入数据和标签
                batch_xs_input = batch_input_wl[:, :-2]
                batch_xs_label = batch_input_wl[:, -2:]

                # 获取下一个批次的数据作为比较
                offset = ((i + 1) * batch_size) % (n_samples - batch_size)
                batch_ben_input_s = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_mal_input_s = mal_data_combined[offset:(
                    offset + batch_size), :]
                batch_ben_input_s_cpu = batch_ben_input_s.cpu().numpy()
                batch_mal_input_s_cpu = batch_mal_input_s.cpu().numpy()
                batch_input = np.row_stack(
                    (batch_ben_input_s_cpu, batch_mal_input_s_cpu))

                batch_input_wl = np.column_stack((batch_input, batch_label))
                np.random.shuffle(batch_input_wl)  # 打乱合并后的数据

                # 分离输入数据和标签
                batch_xcomp_input = batch_input_wl[:, :-2]
                batch_xcomp_label = batch_input_wl[:, -2:]

                # 获取目标数据
                batch_xs_target = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_xs_target = torch.cat(
                    (batch_xs_target, batch_xs_target), dim=0)
                assert batch_xs_input.shape == batch_xs_target.shape

                # 清零之前的梯度
                optimizer.zero_grad()

                if isinstance(batch_xs_input, np.ndarray):
                    batch_xs_input = torch.tensor(
                        batch_xs_input, dtype=torch.float32).to(device)

                if isinstance(batch_xs_target, np.ndarray):
                    batch_xs_target = torch.tensor(
                        batch_xs_target, dtype=torch.float32).to(device)

                if isinstance(batch_xs_label, np.ndarray):
                    batch_xs_label = torch.tensor(
                        batch_xs_label, dtype=torch.float32).to(device)

                if isinstance(batch_xcomp_label, np.ndarray):
                    batch_xcomp_label = torch.tensor(
                        batch_xcomp_label, dtype=torch.float32).to(device)

                if isinstance(batch_xcomp_input, np.ndarray):
                    batch_xcomp_input = torch.tensor(
                        batch_xcomp_input, dtype=torch.float32).to(device)

                self.x_hat = batch_xs_input
                self.x = batch_xs_target
                self.x_comp = batch_xcomp_input
                self.label_x = batch_xs_label
                self.label_x_comp = batch_xcomp_label
                self.keep_prob = 0.9

                # 前向传播
                y, z, loss, marginal_likelihood, KL_divergence = self.forward(batch_xs_input,
                                                                              batch_xs_target,
                                                                              batch_xcomp_input,
                                                                              batch_xs_label,
                                                                              batch_xcomp_label,
                                                                              0.9)

                # 反向传播
                loss.backward()

                # 更新权重
                optimizer.step()

            # 打印每个周期的损失信息
            logger.info(
                f"epoch[{epoch}/{n_epochs}]: L_tot {loss.item():.2f} L_likelihood {marginal_likelihood.item():.2f} L_divergence {KL_divergence.item():.2f}")

        # 检查模型保存路径是否存在，如果不存在，则创建
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

        # 保存当前的模型参数
        torch.save(self.state_dict(), self.model_save_path)

        # 如果开启了详细输出模式，显示模型保存路径
        if verbose:
            print(f'模型保存在路径： {self.model_save_path}')

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))


class VAE_SU(nn.Module, DetectorTemplate):
    def __init__(self,
                 input_size=10000,
                 n_hidden=200,
                 n_epochs=50,
                 z_dim=20,
                 learn_rate=1e-3,
                 Loss_type='1',
                 KLW=10,
                 NLOSSW=10,
                 name='fd_vae',
                 device='cuda:0',
                 **kwargs):

        super(VAE_SU, self).__init__()
        DetectorTemplate.__init__(self)

        # Remove kwargs as it seems unused
        del kwargs

        self.nb_classes = 2

        self.hparams = locals()
        self.device = device
        self.name = name
        self.tau = 500

        # Initialize and build the MLP model
        self.Mlp = mu_sigma_MLP(num_epoch=n_epochs,
                                learn_rate=learn_rate,
                                z_dim=z_dim,
                                name=name
                                )

        # Initialize and build the VAE model
        self.Vae = VAE_2(dim_img=input_size,
                         n_hidden=n_hidden,
                         dim_z=z_dim,
                         KLW=KLW,
                         NLOSSW=NLOSSW,
                         loss_type=Loss_type,
                         learn_rate=1e-3,
                         name=name
                         )

        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'fd_vae') + '_' + self.name,
                                         'model.pth')

        # 日志中打印模型的结构信息
        logger.info(
            '=====================================fd_vae model architecture=============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

        self.dim = self.Vae.dim_img

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau  # 返回tau，即决策阈值

    def forward(self, x, a=None, **kwargs):
        self.Vae.load()
        self.Mlp.load()

        # check_requires_grad(self.Vae)
        # check_requires_grad(self.Mlp)

        x = x.float()
        y, muvae, sigmavae = self.Vae.f(x)
        outputs = self.Mlp(muvae, sigmavae)
        loss_reconstruction = ((y - x) ** 2).sum(dim=-1)

        x_cent = torch.softmax(outputs, dim=-1)
        # print("x_cent", x_cent)

        return x_cent, loss_reconstruction

    def fit(self, train_data_producer, verbose=True):
        # train Vae
        vae_model = self.Vae
        self.Vae.train_model(train_data_producer, device=self.device)

        # train Mlp
        self.Mlp.train_model(
            vae_model, train_data_producer, device=self.device)

        # 检查模型保存路径是否存在，如果不存在，则创建
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        # 保存当前的模型参数
        torch.save(self.state_dict(), self.model_save_path)
        # 如果开启了详细输出模式，显示模型保存路径
        if verbose:
            print(f'模型保存在路径： {self.model_save_path}')

    def load(self):
        """
        从磁盘加载模型参数
        """
        self.Vae.load_state_dict(torch.load(self.model_save_path))
        self.Mlp.load_state_dict(torch.load(self.model_save_path))

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []  # 存储每批数据的真实标签

        self.Vae.load()
        self.Mlp.load()

        # check_requires_grad(self.Vae)
        # check_requires_grad(self.Mlp)

        with torch.no_grad():
            for x, l in test_data_producer:
                x, l = utils.to_device(x, l, self.device)

                x_cent, loss_reconstruction = self.forward(x)
                y_cent.append(x_cent)
                x_prob.append(loss_reconstruction)
                # Store the actual labels instead of the VAE output
                gt_labels.append(l)

        # 将所有批次的置信度垂直堆叠成一个张量
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)

        # 将所有批次的真实标签连接成一个张量
        gt_labels = torch.cat(gt_labels, dim=0)

        return y_cent, x_prob, gt_labels

    # 仅支持恶意软件样本的批量推理

    def inference_batch_wise(self, x):
        assert isinstance(x, torch.Tensor)
        x = x.float()
        self.Vae.load()
        self.Mlp.load()

        y, muvae, sigmavae = self.Vae.f(x)
        outputs = self.Mlp(muvae, sigmavae)

        # print(outputs.shape)

        x_cent_values = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
        x_cent = x_cent_values[:2] if len(
            x_cent_values.shape) == 1 else x_cent_values[:, :2]

        loss_reconstruction = ((y - x) ** 2).sum(dim=-1)
        reloss = loss_reconstruction.detach().cpu().numpy()

        return x_cent, reloss

    def indicator(self, reloss, y_pred=None):
        # 手动设置metric
        metric = 500
        if isinstance(reloss, np.ndarray):
            reloss = torch.tensor(reloss, device=self.device)
            metric_tensor = torch.tensor(metric).to(reloss.device)
            return (reloss <= metric_tensor).cpu().numpy()
        elif isinstance(reloss, torch.Tensor):
            return reloss <= metric
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # 预测标签并进行评估

    def predict(self, test_data_producer, indicator_masking=True, metric=5000):
        y_cent, reloss, y_true = self.inference(test_data_producer)

        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        # 获取指示器标志，用于后续的遮蔽或过滤
        indicator_flag = self.indicator(reloss, metric).cpu().numpy()

        # 定义一个内部函数来评估模型的性能
        def measurement(_y_true, _y_pred):
            # 导入所需的评估指标库
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score

            # 计算并打印准确率和平衡准确率
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info(
                f"The accuracy on the test dataset is {accuracy * 100:.5f}%")
            logger.info(
                f"The balanced accuracy on the test dataset is {b_accuracy * 100:.5f}%")

            # 检查是否所有类都存在于真实标签中
            if np.any([np.all(_y_true == i) for i in range(self.nb_classes)]):
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

        # 再次进行评估
        measurement(y_true, y_pred)

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))
