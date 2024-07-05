"""
max adversarial training framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import random
import time

import torch
import torch.optim as optim
import numpy as np

from core.attack.max import Max
from core.attack.stepwise_max import StepwiseMax
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.max_adv_training')
logger.addHandler(ErrorHandler)


class MaxAdvTraining(object):
    """
    对模型进行对抗训练，包含多种攻击的混合

    Parameters
    -------
    @param model: Object, 需要被保护的模型，例如 MalwareDetector
    @attack_model: Object, 用于在特征空间上生成对抗性恶意软件的对手模型
    """

    def __init__(self, model, attack=None, attack_param=None):
        """
        初始化函数

        :param model: 需要进行对抗训练的模型
        :param attack: 使用的攻击方式，默认为 None
        :param attack_param: 攻击参数，默认为 None
        """
        self.model = model  # 要进行对抗训练的模型
        # 如果传入了攻击方式，则检查该攻击方式是否是 Max 或 StepwiseMax 的实例
        if attack is not None:
            assert isinstance(attack, (Max, StepwiseMax))
            # 检查攻击方式是否包含 'is_attacker' 属性，如果有，确保它的值为 False
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker
        self.attack = attack  # 所使用的攻击方式
        self.attack_param = attack_param  # 攻击的参数

        self.name = self.model.name  # 获取模型的名称
        # 设置模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'md_at_ma') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path  # 更新模型内部的保存路径
        # 在日志中记录所使用的对抗攻击的类型
        logger.info("Adversarial training incorporating the attack {}".format(
            type(self.attack).__name__))

    # 这个fit方法使用对抗训练策略来训练恶意软件检测模型。其主要流程包括在每个训练迭代中产生对抗样本，
    # 然后用这些对抗样本来训练模型。验证部分则是用来评估模型在对抗样本上的性能，并据此选择最佳模型。
    def fit(self, train_data_producer, validation_data_producer=None, adv_epochs=50,
            beta=0.01,
            lr=0.001,
            under_sampling_ratio=1.,
            weight_decay=5e-0, verbose=True):
        """
        使用对抗训练增强恶意软件检测器。

        参数
        -------
        train_data_producer: Object, 用于生产一批训练数据的dataloader对象
        validation_data_producer: Object, 用于生产验证数据集的dataloader对象
        adv_epochs: Integer, 对抗训练的迭代次数
        beta: Float, 对抗损失的惩罚因子
        lr: Float, Adam优化器的学习率
        under_sampling_ratio: [0,1], 对恶意软件样本进行下采样的比例，用于对抗训练
        weight_decay: Float, 权重衰减，惩罚因子，默认值为 5e-4
        verbose: Boolean, 是否显示详细信息
        """
        # 初始化优化器
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=weight_decay)
        total_time = 0.
        nbatches = len(train_data_producer)

        logger.info("Max对抗训练开始 ...")

        # 初始化最佳验证准确率、对抗准确率和最佳轮次
        best_acc_val = 0.
        acc_val_adv_be = 0.
        best_epoch = 0

        # 开始对抗性训练
        for i in range(adv_epochs):
            # 设置随机种子以保证结果的可复现性
            random.seed(0)

            # 初始化每轮的损失和准确率列表
            losses, accuracies = [], []

            # 遍历训练数据
            for idx_batch, (x_batch, y_batch) in enumerate(train_data_producer):
                # 将数据转换为适合模型的张量格式
                x_batch, y_batch = utils.to_tensor(
                    x_batch.double(), y_batch.long(), self.model.device)

                # 获取当前批次的数据量
                batch_size = x_batch.shape[0]

                # 获取恶意和正常数据
                mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag = \
                    utils.get_mal_ben_data(x_batch, y_batch)

                # 如果指定了下采样比率，对恶意数据进行下采样
                if 0. < under_sampling_ratio < 1.:
                    n_mal = mal_x_batch.shape[0]
                    n_mal_sampling = int(
                        under_sampling_ratio * n_mal) if int(under_sampling_ratio * n_mal) > 1 else 1
                    idx_sampling = random.sample(range(n_mal), n_mal_sampling)
                    mal_x_batch, mal_y_batch = mal_x_batch[idx_sampling], mal_y_batch[idx_sampling]

                # 如果没有获取到恶意或正常数据，则跳过当前循环
                if null_flag:
                    continue

                # 开始对抗性攻击，首先设置模型为验证模式
                start_time = time.time()
                self.model.eval()

                # 执行对抗性攻击
                pertb_mal_x = self.attack.perturb(
                    self.model, mal_x_batch, mal_y_batch, **self.attack_param)
                pertb_mal_x = utils.round_x(pertb_mal_x, 0.5)
                total_time += time.time() - start_time

                # 合并原始数据和受到攻击的数据
                x_batch = torch.cat([x_batch, pertb_mal_x], dim=0)
                y_batch = torch.cat([y_batch, mal_y_batch])

                # 设置模型为训练模式，并进行前向传播和损失计算
                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                logits = self.model.forward(x_batch)

                # 计算损失，分为原始数据损失和对抗数据损失
                loss_train = self.model.customize_loss(
                    logits[:batch_size], y_batch[:batch_size])
                loss_train += beta * \
                    self.model.customize_loss(
                        logits[batch_size:], y_batch[batch_size:])

                # 反向传播和参数更新
                loss_train.backward()
                optimizer.step()
                total_time += time.time() - start_time

                # 计算训练的准确率
                acc_train = (logits.argmax(1) == y_batch).sum(
                ).item() / x_batch.size()[0]
                accuracies.append(acc_train)
                losses.append(loss_train.item())

                # 如果需要，打印详细的训练信息
                if verbose:
                    mins, secs = int(total_time / 60), int(total_time % 60)
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}%.')

            # 打印每轮训练的平均损失和准确率
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')

            # 选择模型
            self.model.eval()

            # 考虑到模型训练时间可能很长，为了防止意外中断，暂时保存模型
            self.save_to_disk(i + 1, optimizer, self.model_save_path + '.tmp')

            # 设置模型为攻击者模式，以进行对抗性验证
            self.attack.is_attacker = True

            # 初始化验证结果和准确率列表
            res_val = []
            avg_acc_val = []

            # 遍历验证数据
            for x_val, y_val in validation_data_producer:
                # 将数据转换为适合模型的张量格式
                x_val, y_val = utils.to_tensor(
                    x_val.double(), y_val.long(), self.model.device)

                # 让模型前向传播并得到结果
                logits = self.model.forward(x_val)

                # 计算当前批次的准确率，并加入到平均准确率列表中
                acc_val = (logits.argmax(1) == y_val).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)

                # 从验证集中获取恶意软件数据
                mal_x_batch, mal_y_batch, null_flag = utils.get_mal_data(
                    x_val, y_val)

                # 如果没有恶意软件数据，则跳过当前循环
                if null_flag:
                    continue

                # 对模型进行对抗性攻击
                pertb_mal_x = self.attack.perturb(
                    self.model, mal_x_batch, mal_y_batch, **self.attack_param)
                pertb_mal_x = utils.round_x(pertb_mal_x, 0.5)

                # 对受到攻击的数据进行模型推断
                y_cent_batch, x_density_batch = self.model.inference_batch_wise(
                    pertb_mal_x)

                # 获取预测结果
                y_pred = np.argmax(y_cent_batch, axis=-1)

                # 判断预测结果是否为恶意软件，并加入到验证结果列表中
                res_val.append(y_pred == 1.)

            # 确保有有效的验证结果
            assert len(res_val) > 0
            res_val = np.concatenate(res_val)

            # 计算对抗性攻击下的准确率
            acc_val_adv = np.sum(res_val).astype(float) / res_val.shape[0]

            # 计算总体验证准确率（正常准确率和对抗性准确率的平均值）
            acc_val = (np.mean(avg_acc_val) + acc_val_adv) / 2.

            # 由于我们每个epoch后都会寻找新的阈值，这可能会妨碍训练的收敛。
            # 我们只保存最后几个epoch的模型参数，因为这时可能已经得到了一个训练良好的模型。
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                acc_val_adv_be = acc_val_adv
                best_epoch = i + 1

                # 保存当前最佳的模型
                self.save_to_disk(best_epoch, optimizer, self.model_save_path)

            if verbose:
                logger.info(
                    f"\tVal accuracy {acc_val * 100:.4}% with accuracy {acc_val_adv * 100:.4}% under attack.")
                logger.info(
                    f"\tModel select at epoch {best_epoch} with validation accuracy {best_acc_val * 100:.4}% and accuracy {acc_val_adv_be * 100:.4}% under attack.")
                if hasattr(self.model, 'tau'):
                    logger.info(
                        f'The threshold is {self.model.tau}.'
                    )
            self.attack.is_attacker = False

    def load(self):
        # 断言模型的保存路径是否存在，如果不存在则表示需要首先训练模型
        assert path.exists(self.model_save_path), 'train model first'

        # 从保存的路径中加载模型
        ckpt = torch.load(self.model_save_path)

        # 加载整体模型的权重
        self.model.load_state_dict(ckpt['model'])

    def save_to_disk(self, epoch, optimizer, save_path=None):
        # 如果指定的保存路径不存在，创建对应的文件夹
        if not path.exists(save_path):
            utils.mkdir(path.dirname(save_path))

        # 保存整体模型的权重、当前轮次和优化器的状态
        torch.save({'model': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()
                    },
                   save_path)
