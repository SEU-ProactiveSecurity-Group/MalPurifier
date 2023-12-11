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


class AMalwareDetectionPAD(object):
    """
    针对恶意软件检测的对抗训练类，整合了“max”或逐步“max”攻击

    参数
    -------
    model: Object, 需要被保护的模型，例如 MalwareDetector
    attack_model: Object, 在特征空间上生成对抗性恶意软件的对手模型
    """

    def __init__(self, model, attack=None, attack_param=None):
        # 将传入的模型赋值给当前对象的model属性
        self.model = model
        
        # 断言，确保传入的模型有forward_g这个属性或方法
        assert hasattr(self.model, 'forward_g')

        # 如果传入了attack参数（也就是攻击模型）
        if attack is not None:
            # 断言，确保传入的攻击模型是Max或StepwiseMax其中之一
            assert isinstance(attack, (Max, StepwiseMax))
            
            # 如果攻击模型的字典属性中有'is_attacker'这个key
            if 'is_attacker' in attack.__dict__.keys():
                
                # 断言，确保attack的is_attacker属性值为False
                assert not attack.is_attacker
        
        # 将传入的attack赋值给当前对象的attack属性
        self.attack = attack
        
        # 将传入的attack_param赋值给当前对象的attack_param属性
        self.attack_param = attack_param

        # 将模型的name属性值赋值给当前对象的name属性
        self.name = self.model.name
        
        # 定义模型的保存路径
        self.model_save_path = path.join(config.get('experiments', 'amd_pad_ma') + '_' + self.name, 'model.pth')
        
        # 将定义好的模型保存路径赋值给模型对象的model_save_path属性
        self.model.model_save_path = self.model_save_path


    def fit(self, train_data_producer, validation_data_producer=None, adv_epochs=50,
            beta_1=0.1, beta_2=1, lmda_lower_bound=1e-3, lmda_upper_bound=1e3, 
            use_continuous_pert=True, lr=0.001, under_sampling_ratio=1., 
            weight_decay=5e-0, verbose=True):
        """
        应用对抗训练来增强恶意软件检测器。

        参数
        -------
        train_data_producer: Object, 用于生成训练数据批次的数据加载器对象
        validation_data_producer: Object, 用于生成验证数据集的数据加载器对象
        adv_epochs: Integer, 对抗训练的迭代次数
        beta_1: Float, 对抗损失的惩罚因子
        beta_2: Float, 对抗损失的惩罚因子
        lmda_lower_bound: Float, 惩罚因子的下界
        lmda_upper_bound: Float, 惩罚因子的上界
        use_continuous_pert: Boolean, 是否使用连续扰动
        lr: Float, Adam优化器的学习率
        under_sampling_ratio: [0,1], 对抗训练中恶意软件示例的下采样比率
        weight_decay: Float, 惩罚因子，默认值为GAT(Graph ATtention layer)中的5e-4
        verbose: Boolean, 是否显示详细信息
        """
        # 定义一个非负的权重约束
        constraint = utils.NonnegWeightConstraint()
        
        # 定义优化器为Adam
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_time = 0.
        
        # 计算训练数据批次的数量
        nbatches = len(train_data_producer)
        
        # 根据给定的惩罚因子上下界，生成一个惩罚因子空间
        lmda_space = np.logspace(np.log10(lmda_lower_bound),
                                np.log10(lmda_upper_bound),
                                num=int(np.log10(lmda_upper_bound / lmda_lower_bound)) + 1)
        
        # 日志输出开始进行最大对抗训练
        logger.info("Max adversarial training is starting ...")
        best_acc_val = 0.
        acc_val_adv_be = 0.
        best_epoch = 0

        # 对抗训练的主循环
        for i in range(adv_epochs):
            random.seed(0)
            losses, accuracies = [], []

            # 从数据生产者中迭代获取数据批次
            for idx_batch, (x_batch, y_batch) in enumerate(train_data_producer):
                
                # 将数据转换为模型所需的格式
                x_batch, y_batch = utils.to_tensor(x_batch.double(), y_batch.long(), self.model.device)
                batch_size = x_batch.shape[0]

                # 为对手检测器添加椒盐噪声
                x_batch_noises = torch.clamp(x_batch + utils.psn(x_batch, np.random.uniform(0, 0.5)), min=0., max=1.)
                
                # 连接原始数据和噪声数据
                x_batch_ = torch.cat([x_batch, x_batch_noises], dim=0)
                y_batch_ = torch.cat([torch.zeros(batch_size, ), torch.ones(batch_size, )]).long().to(self.model.device)
                
                # 打乱数据的顺序
                idx = torch.randperm(y_batch_.shape[0])
                x_batch_ = x_batch_[idx]
                y_batch_ = y_batch_[idx]

                # 为分类器获取恶意和良性数据
                mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag = \
                    utils.get_mal_ben_data(x_batch, y_batch)
                
                # 如果设置了下采样比率，则进行下采样
                if 0. < under_sampling_ratio < 1.:
                    n_mal = mal_x_batch.shape[0]
                    n_mal_sampling = int(under_sampling_ratio * n_mal) if int(under_sampling_ratio * n_mal) > 1 else 1
                    idx_sampling = random.sample(range(n_mal), n_mal_sampling)
                    mal_x_batch, mal_y_batch = mal_x_batch[idx_sampling], mal_y_batch[idx_sampling]
                
                # 如果数据为空，则跳过这一批次
                if null_flag:
                    continue

                start_time = time.time()

                # 攻击方法通过使用不同的超参数lambda扰动特征向量，目的是尽可能获得对抗样本
                self.model.eval() # 将模型设置为评估模式
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  min_lambda_=np.random.choice(lmda_space), 
                                                  # 当lambda值较小时，我们无法进行有效的攻击
                                                  max_lambda_=lmda_upper_bound,
                                                  **self.attack_param
                                                  )
                
                # 将扰动后的数据四舍五入
                disc_pertb_mal_x_ = utils.round_x(pertb_mal_x, 0.5)
                total_time += time.time() - start_time
                
                # 将原始数据和扰动后的数据进行拼接
                x_batch = torch.cat([x_batch, disc_pertb_mal_x_], dim=0)
                y_batch = torch.cat([y_batch, mal_y_batch])

                # 如果使用连续扰动
                if use_continuous_pert:
                    filter_flag = torch.amax(torch.abs(pertb_mal_x - mal_x_batch), dim=-1) <= 1e-6
                    pertb_mal_x = pertb_mal_x[~filter_flag]
                    orgin_mal_x = mal_x_batch[~filter_flag]
                    x_batch_ = torch.cat([x_batch_, orgin_mal_x, pertb_mal_x], dim=0)
                    n_pertb_mal = pertb_mal_x.shape[0]
                else: # 否则
                    filter_flag = torch.sum(torch.abs(disc_pertb_mal_x_ - mal_x_batch), dim=-1) == 0
                    disc_pertb_mal_x_ = disc_pertb_mal_x_[~filter_flag]
                    orgin_mal_x = mal_x_batch[~filter_flag]
                    x_batch_ = torch.cat([x_batch_, orgin_mal_x, disc_pertb_mal_x_], dim=0)
                    n_pertb_mal = disc_pertb_mal_x_.shape[0]

                y_batch_ = torch.cat([y_batch_, torch.zeros((n_pertb_mal * 2,), ).to(
                    self.model.device)]).double()
                y_batch_[-n_pertb_mal:] = 1.
                start_time = time.time()

                self.model.train() # 将模型设置为训练模式
                optimizer.zero_grad() # 清除之前的梯度
                logits_f = self.model.forward_f(x_batch) # 通过模型的forward_f进行预测
                logits_g = self.model.forward_g(x_batch_) # 通过模型的forward_g进行预测

                # 计算训练损失
                loss_train = self.model.customize_loss(logits_f[:batch_size],
                                                       y_batch[:batch_size],
                                                       logits_g[:2 * batch_size],
                                                       y_batch_[:2 * batch_size])
                
                loss_train += self.model.customize_loss(logits_f[batch_size:],
                                                        y_batch[batch_size:],
                                                        logits_g[2 * batch_size:],
                                                        y_batch_[2 * batch_size:],
                                                        beta_1=beta_1,
                                                        beta_2=beta_2
                                                        )


                # 对损失进行反向传播
                loss_train.backward()
                
                # 更新模型的参数
                optimizer.step()
                
                # 对具有'non_neg_layer'名字的模型层进行裁剪，以满足非负约束
                for name, module in self.model.named_modules():
                    if 'non_neg_layer' in name:
                        module.apply(constraint)

                # 计算累计的训练时间
                total_time += time.time() - start_time
                mins, secs = int(total_time / 60), int(total_time % 60)

                # 计算训练的准确率
                acc_f_train = (logits_f.argmax(1) == y_batch).sum().item()
                acc_f_train /= x_batch.size()[0]
                accuracies.append(acc_f_train)
                losses.append(loss_train.item())
                
                # 如果开启了详细输出
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                
                # 如果模型有'forward_g'这个属性
                if hasattr(self.model, 'forward_g'):
                    acc_g_train = ((torch.sigmoid(logits_g) >= 0.5) == y_batch_).sum().item()
                    acc_g_train /= x_batch_.size()[0]
                    accuracies.append(acc_g_train)
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}% & {acc_g_train * 100:.2f}%.')
                else:
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}%.')

                # 输出整个训练周期的损失和准确率
                if verbose:
                    logger.info(
                        f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')

            # 临时保存模型
            self.save_to_disk(self.model_save_path + '.tmp', i + 1, optimizer)

            # 设置模型为评估模式
            self.model.eval()
            self.attack.is_attacker = True
            res_val = []
            avg_acc_val = []

            # 在验证数据集上进行验证
            for x_val, y_val in validation_data_producer:
                # 将数据转换为tensor并送入相应的设备上
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.model.device)

                # 进行前向传播并计算准确率
                logits_f = self.model.forward_f(x_val)
                acc_val = (logits_f.argmax(1) == y_val).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)

                # 获取恶意数据
                mal_x_batch, mal_y_batch, null_flag = utils.get_mal_data(x_val, y_val)
                if null_flag:
                    continue
                # 对恶意数据进行扰动
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  min_lambda_=1e-5,
                                                  max_lambda_=1e5,
                                                  **self.attack_param
                                                  )
                # 使用模型进行预测
                y_cent_batch, x_density_batch = self.model.inference_batch_wise(pertb_mal_x)
                if hasattr(self.model, 'indicator'):
                    indicator_flag = self.model.indicator(x_density_batch)
                else:
                    indicator_flag = np.ones([x_density_batch.shape[0], ]).astype(np.bool)
                y_pred = np.argmax(y_cent_batch, axis=-1)
                res_val.append((~indicator_flag) | ((y_pred == 1.) & indicator_flag))

            # 确保有验证结果
            assert len(res_val) > 0
            res_val = np.concatenate(res_val)
            acc_val_adv = np.sum(res_val).astype(float) / res_val.shape[0]
            acc_val = (np.mean(avg_acc_val) + acc_val_adv) / 2.

            # 如果当前模型的验证准确率超过之前的最佳结果，则保存模型参数
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                acc_val_adv_be = acc_val_adv
                best_epoch = i + 1
                self.save_to_disk(self.model_save_path)

            # 如果开启了详细输出
            if verbose:
                logger.info(
                    f"\tVal accuracy {acc_val * 100:.4}% with accuracy {acc_val_adv * 100:.4}% under attack.")
                logger.info(
                    f"\tModel select at epoch {best_epoch} with validation accuracy {best_acc_val * 100:.4}% and accuracy {acc_val_adv_be * 100:.4}% under attack.")

            self.attack.is_attacker = False


    def load(self):
        # 确保模型保存路径存在，如果不存在则表示需要先训练模型
        assert path.exists(self.model_save_path), 'train model first'
        # 加载模型
        ckpt = torch.load(self.model_save_path)
        self.model.load_state_dict(ckpt['model'])

    def save_to_disk(self, save_path, epoch=None, optimizer=None):
        # 如果保存路径不存在，则创建对应的目录
        if not path.exists(path.dirname(save_path)):
            utils.mkdir(path.dirname(save_path))
        # 如果给定了训练周期(epoch)和优化器(optimizer)，则将模型、训练周期、优化器状态都保存
        if epoch is not None and optimizer is not None:
            torch.save({'model': self.model.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                        },
                       save_path)
        # 否则，只保存模型
        else:
            torch.save({'model': self.model.state_dict()}, save_path)
