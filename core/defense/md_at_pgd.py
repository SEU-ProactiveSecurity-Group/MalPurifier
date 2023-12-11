"""
adversarial training incorporating pgd linf attack
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

import torch
import torch.optim as optim
import numpy as np

from core.attack.pgd import PGD
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.adv_pgd_linf')
logger.addHandler(ErrorHandler)


# 这份代码是关于PGD（Projected Gradient Descent，投影梯度下降）对抗训练的类。
# 这种对抗训练方法使用了PGD攻击，可以增强模型对于对抗性攻击的鲁棒性。
class PGDAdvTraining(object):
    """PGD对抗训练

    参数
    -------
    @param model, 对象, 需要被保护的模型, 如 MalwareDetector
    @attack_model: 对象, 用于在特征空间上生成对抗恶意软件的对手模型
    """

    def __init__(self, model, attack=None, attack_param=None):
        # 初始化被保护的模型
        self.model = model
        
        # 如果提供了攻击模型，则检查它是否是PGD类型
        if attack is not None:
            assert isinstance(attack, PGD)
            # 确保attack对象的属性中不包含'is_attacker'或该属性的值为False
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker
                
        # 初始化攻击模型和攻击参数
        self.attack = attack
        self.attack_param = attack_param

        # 模型的名称
        self.name = self.model.name
        # 设置模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'md_at_pgd') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path
        # 日志输出：正在进行的对抗训练攻击类型
        logger.info("Adversarial training incorporating the attack {}".format(type(self.attack).__name__))


    def fit(self, train_data_producer, validation_data_producer=None, epochs=5, adv_epochs=45,
            beta=0.001,
            lr=0.005,
            weight_decay=5e-0, verbose=True):
        """
        使用对抗训练来增强恶意软件检测器的性能。

        参数
        -------
        @param train_data_producer: 对象, 用于生成一批训练数据的dataloader对象
        @param validation_data_producer: 对象, 用于生成验证数据集的dataloader对象
        @param epochs: 整数, 对抗训练的轮次
        @param adv_epochs: 整数, 对抗训练的轮次
        @param beta: 浮点数, 对抗损失的惩罚因子
        @param lr: 浮点数, Adam优化器的学习率
        @param weight_decay: 浮点数, 惩罚因子，默认值为5e-4
        @param verbose: 布尔值, 是否显示详细信息
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)  # 定义优化器
        total_time = 0.  # 记录总训练时间
        nbatches = len(train_data_producer)  # 计算数据批次数
        logger.info("对抗训练开始 ...")
        best_acc_val = 0.  # 最佳验证准确率
        acc_val_adv_be = 0.  
        best_epoch = 0  # 最佳轮次
        for i in range(adv_epochs):
            losses, accuracies = [], []  # 记录每轮的损失和准确率
            for idx_batch, (x_batch, y_batch) in enumerate(train_data_producer):
                x_batch, y_batch = utils.to_tensor(x_batch.double(), y_batch.long(), self.model.device)  # 数据转为张量
                batch_size = x_batch.shape[0]
                # 分割数据为恶意和良性
                mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag = \
                    utils.get_mal_ben_data(x_batch, y_batch)
                if null_flag:
                    continue
                start_time = time.time()
                self.model.eval()  # 设置模型为评估模式
                
                # ⭐ 对数据进行对抗扰动
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                **self.attack_param
                                                )
                total_time += time.time() - start_time
                
                # 合并对抗样本和原始样本
                x_batch = torch.cat([ben_x_batch, pertb_mal_x], dim=0)
                y_batch = torch.cat([ben_y_batch, mal_y_batch])
                start_time = time.time()
                self.model.train()  # 设置模型为训练模式
                optimizer.zero_grad()  # 清零梯度
                logits = self.model.forward(x_batch)  # 前向传播
                loss_train = self.model.customize_loss(logits,
                                                    y_batch)  # 计算损失

                loss_train.backward()  # 反向传播
                optimizer.step()  # 更新参数

                total_time += time.time() - start_time
                mins, secs = int(total_time / 60), int(total_time % 60)  # 计算训练时间
                acc_train = (logits.argmax(1) == y_batch).sum().item()  # 计算训练准确率
                acc_train /= x_batch.size()[0]
                accuracies.append(acc_train)  # 记录准确率
                losses.append(loss_train.item())  # 记录损失
                if verbose:
                    logger.info(
                        f'小批次: {i * nbatches + idx_batch + 1}/{adv_epochs * nbatches} | 已训练时间 {mins:.0f} 分钟, {secs} 秒.')
                    logger.info(
                        f'训练损失（小批次级别）: {losses[-1]:.4f} | 训练准确率: {acc_train * 100:.2f}%.')
            if verbose:
                logger.info(
                    f'训练损失（轮次级别）: {np.mean(losses):.4f} | 训练准确率: {np.mean(accuracies) * 100:.2f}')

            # 将模型设置为评估模式
            # 这段代码描述了在每个epoch结束后，使用验证集来评估模型的性能。
            # 并在遭受对抗攻击时，检查模型的准确性。如果当前模型的验证准确率超过之前的最佳准确率，就保存模型。
            self.model.eval()

            # 因为训练可能会很耗时（为了防止意外中断，临时保存模型）
            self.save_to_disk(i + 1, optimizer, self.model_save_path + '.tmp')

            # 用于存储验证结果和平均验证准确率
            res_val = []
            avg_acc_val = []

            # 遍历验证数据集
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.model.device)
                logits = self.model.forward(x_val)
                acc_val = (logits.argmax(1) == y_val).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)

                # 获取验证数据集中的恶意数据
                mal_x_batch, mal_y_batch, null_flag = utils.get_mal_data(x_val, y_val)
                if null_flag:
                    continue

                # 对验证数据的恶意部分进行对抗扰动
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  **self.attack_param
                                                  )

                # 对扰动后的数据进行预测
                y_cent_batch, x_density_batch = self.model.inference_batch_wise(pertb_mal_x)
                y_pred = np.argmax(y_cent_batch, axis=-1)
                res_val.append(y_pred == 1.)

            # 确保有验证结果
            assert len(res_val) > 0
            res_val = np.concatenate(res_val)

            # 计算对抗样本下的准确率
            acc_val_adv = np.sum(res_val).astype(float) / res_val.shape[0]

            # 计算总体验证准确率
            acc_val = (np.mean(avg_acc_val) + acc_val_adv) / 2.

            # 因为我们在每个epoch之后寻找一个新的阈值，这可能会阻碍训练的收敛。
            # 所以在最后几个epoch保存模型参数，因为可能会得到一个训练良好的模型。
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                acc_val_adv_be = acc_val_adv
                best_epoch = i + 1
                self.save_to_disk(best_epoch, optimizer, self.model_save_path)

            # 如果verbose为True，打印验证相关信息
            if verbose:
                logger.info(
                    f"\t验证准确率 {acc_val * 100:.4}%，在攻击下的准确率为 {acc_val_adv * 100:.4}%。")
                logger.info(
                    f"\t在第 {best_epoch} 轮选择的模型，验证准确率为 {best_acc_val * 100:.4}%，在攻击下的准确率为 {acc_val_adv_be * 100:.4}%。")
                # 如果模型有阈值属性，打印阈值
                if hasattr(self.model, 'tau'):
                    logger.info(
                        f'当前阈值为 {self.model.tau}。'
                    )

    # 这段代码定义了两个方法：load 和 save_to_disk。

    # load 方法用于加载预训练的模型权重。首先，它会检查模型的保存路径是否存在。
    # 如果存在，则加载模型权重；否则，提示用户首先训练模型。
    def load(self):
        # 断言检查模型存储路径是否存在，如果不存在则提示先进行训练
        assert path.exists(self.model_save_path), 'train model first'

        # 加载模型权重
        ckpt = torch.load(self.model_save_path)
        self.model.load_state_dict(ckpt['model'])


    # save_to_disk 方法用于将模型权重、当前的epoch和优化器状态保存到磁盘上的指定路径。
    # 如果保存路径不存在，它会首先创建这个路径。然后，使用PyTorch的 torch.save 方法将数据保存到指定路径。
    def save_to_disk(self, epoch, optimizer, save_path=None):
        # 检查保存路径是否存在，如果不存在则创建相应的文件夹
        if not path.exists(save_path):
            utils.mkdir(path.dirname(save_path))
        
        # 保存模型权重、当前的epoch和优化器状态到指定路径
        torch.save({'model': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()
                    },
                   save_path)

