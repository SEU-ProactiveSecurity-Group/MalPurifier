import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.max')
logger.addHandler(ErrorHandler)
EXP_OVER_FLOW = 1e-30


class Max(BaseAttack):
    """
    Max攻击：迭代地从多个攻击方法中选择结果。

    参数
    --------
    @param attack_list: List, 已实例化的攻击对象的列表。
    @param varepsilon: Float, 用于判断收敛性的标量。
    """

    def __init__(self, attack_list, varepsilon=1e-20,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        """
        构造函数

        参数:
        - attack_list: 已实例化的攻击对象的列表，至少应该有一个攻击方法。
        - varepsilon: 用于判断收敛性的标量，默认值为1e-20。
        - is_attacker: Bool, 表示是否为攻击者，默认为True。
        - oblivion: Bool, 一个布尔标志（其功能在这里并未详细说明），默认为False。
        - kappa: Float, 一个浮点数参数，默认为1。
        - manipulation_x: 可能与数据的处理或操纵有关，具体用途未详细说明。
        - omega: 参数omega的具体用途未详细说明。
        - device: 设备，例如'cuda'或'cpu'，用于执行计算。

        注意:
        - 在初始化过程中，会首先检查`attack_list`是否包含至少一个攻击对象。
        """
        super(Max, self).__init__(is_attacker, oblivion, kappa,
                                  manipulation_x, omega, device)  # 调用父类的构造函数
        assert len(attack_list) > 0, '至少需要一个攻击方法。'  # 确保提供了至少一个攻击对象
        self.attack_list = attack_list  # 设置攻击列表
        self.varepsilon = varepsilon  # 设置varepsilon值
        self.device = device  # 设置计算设备

    def perturb(self, model, x, label=None, steps_max=5, min_lambda_=1e-5, max_lambda_=1e5, verbose=False):
        """
        扰动节点特征

        参数
        -----------
        @param model: 受害者模型。
        @param x: torch.FloatTensor, 形状为[batch_size, vocab_dim]的特征向量。
        @param label: torch.LongTensor, 真实标签。
        @param steps_max: Integer, 最大的迭代次数。
        @param min_lambda_: float, 平衡对手检测器的重要性（如果存在）。
        @param max_lambda_: float, 同上。
        @param verbose: Boolean, 是否打印详细日志。

        返回值
        --------
        adv_x: 扰动后的数据。
        """

        # 判断输入数据是否有效
        if x is None or x.shape[0] <= 0:
            return []

        # 将模型设为评估模式，主要是为了禁用一些在训练模式下的特殊层，比如Dropout
        model.eval()

        # 获取输入数据x在当前模型下的损失和完成状态
        with torch.no_grad():
            loss, done = self.get_scores(model, x, label)

        # 存储当前的损失为前一次的损失
        pre_loss = loss

        # 获取输入数据的数量以及其他的维度信息
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        # 初始化攻击样本为输入数据的拷贝
        adv_x = x.detach().clone()

        # 初始化停止标志，用于表示哪些样本已经完成了攻击
        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)

        # 开始主循环，进行多次迭代以改进攻击效果
        for t in range(steps_max):
            # 计算还未完成攻击的样本数量
            num_sample_red = n - torch.sum(stop_flag)

            # 如果所有样本都已完成攻击，结束循环
            if num_sample_red <= 0:
                break

            # 获取那些还未完成攻击的样本的真实标签
            red_label = label[~stop_flag]
            pertbx = []

            # 对于攻击方法列表中的每种攻击方法，尝试对数据进行扰动
            for attack in self.attack_list:
                # 确保每种攻击方法都实现了perturb方法
                assert 'perturb' in type(attack).__dict__.keys()

                # 对于某些特定的攻击方法，在第二次及以后的迭代中取消随机化
                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False

                # 对于名为"Orthogonal"的攻击方法，进行特殊处理
                if 'Orthogonal' in type(attack).__name__:
                    pertbx.append(attack.perturb(
                        model=model, x=adv_x[~stop_flag], label=red_label))
                else:
                    pertbx.append(attack.perturb(model=model, x=adv_x[~stop_flag], label=red_label,
                                                 min_lambda_=1e-5,
                                                 max_lambda_=1e5,
                                                 ))
            # 将所有攻击方法产生的扰动数据合并
            pertbx = torch.vstack(pertbx)

            # 不需要计算梯度，提高计算效率
            with torch.no_grad():
                # 将真实标签复制若干次以匹配所有的攻击列表
                red_label_ext = torch.cat([red_label] * len(self.attack_list))

                # 获取每种攻击方法产生的损失值和成功状态
                loss, done = self.get_scores(model, pertbx, red_label_ext)

                # 调整损失和成功状态的形状以方便后续计算
                loss = loss.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)

                # 判断哪些样本至少有一种攻击方法成功
                success_flag = torch.any(done, dim=-1)

                # 对于没有成功的样本，将其标记为1以进行后续处理
                done[~torch.any(done, dim=-1)] = 1

                # 调整损失值，对于成功的攻击方法，损失值保持不变；对于失败的，损失值变为最小值
                loss = (loss * done.to(torch.float)) + \
                    torch.min(loss) * (~done).to(torch.float)

                # 调整扰动数据的形状以方便后续计算
                pertbx = pertbx.reshape(
                    len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])

                # 选择造成最大损失的扰动数据
                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(
                    num_sample_red), indices]

                # 获取选中的扰动数据的损失值
                a_loss = loss[torch.arange(num_sample_red), indices]

                # 复制当前的停止标志
                pre_stop_flag = stop_flag.clone()

                # 更新停止标志，如果损失值变化很小或者某种攻击方法成功，则停止迭代
                stop_flag[~stop_flag] = (
                    torch.abs(pre_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag

                # 更新前一个损失值
                pre_loss[~pre_stop_flag] = a_loss

            # 如果需要打印日志
            if verbose:
                # 评估最终的扰动数据的成功状态
                with torch.no_grad():
                    _, done = self.get_scores(model, adv_x, label)
                    # 打印攻击成功率
                    logger.info(
                        f"max: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

            # 返回最终的扰动数据
            return adv_x

    def perturb_dae(self, predict_model, purifier, x, label=None, steps_max=5, min_lambda_=1e-5, max_lambda_=1e5, verbose=False, oblivion=False):
        """
        扰动节点特征

        参数
        -----------
        @param model: 受害者模型。
        @param x: torch.FloatTensor, 形状为[batch_size, vocab_dim]的特征向量。
        @param label: torch.LongTensor, 真实标签。
        @param steps_max: Integer, 最大的迭代次数。
        @param min_lambda_: float, 平衡对手检测器的重要性（如果存在）。
        @param max_lambda_: float, 同上。
        @param verbose: Boolean, 是否打印详细日志。

        返回值
        --------
        adv_x: 扰动后的数据。
        """

        # 判断输入数据是否有效
        if x is None or x.shape[0] <= 0:
            return []

        # 将模型设为评估模式，主要是为了禁用一些在训练模式下的特殊层，比如Dropout
        predict_model.eval()
        purifier.eval()

        # 获取输入数据x在当前模型下的损失和完成状态
        with torch.no_grad():
            if not oblivion:
                purified_x = purifier(
                    x.detach().clone().float()).to(torch.double)
            else:
                purified_x = x.detach().clone()
            loss, done = self.get_scores(predict_model, purified_x, label)

        # 存储当前的损失为前一次的损失
        pre_loss = loss

        # 获取输入数据的数量以及其他的维度信息
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        # 初始化攻击样本为输入数据的拷贝
        adv_x = x.detach().clone()

        # 初始化停止标志，用于表示哪些样本已经完成了攻击
        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)

        # 开始主循环，进行多次迭代以改进攻击效果
        for t in range(steps_max):
            # 计算还未完成攻击的样本数量
            num_sample_red = n - torch.sum(stop_flag)

            # 如果所有样本都已完成攻击，结束循环
            if num_sample_red <= 0:
                break

            # 获取那些还未完成攻击的样本的真实标签
            red_label = label[~stop_flag]
            pertbx = []

            # 对于攻击方法列表中的每种攻击方法，尝试对数据进行扰动
            for attack in self.attack_list:
                # 确保每种攻击方法都实现了perturb方法
                assert 'perturb' in type(attack).__dict__.keys()

                # 对于某些特定的攻击方法，在第二次及以后的迭代中取消随机化
                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False

                # 对于名为"Orthogonal"的攻击方法，进行特殊处理
                if 'Orthogonal' in type(attack).__name__:
                    pertbx.append(attack.perturb_dae(predict_model=predict_model, purifier=purifier,
                                  x=adv_x[~stop_flag], label=red_label, oblivion=oblivion))
                else:
                    pertbx.append(attack.perturb_dae(model=predict_model, purifier=purifier, x=adv_x[~stop_flag], label=red_label,
                                                     min_lambda_=1e-5,
                                                     max_lambda_=1e5,
                                                     oblivion=oblivion
                                                     ))

            # 将所有攻击方法产生的扰动数据合并
            pertbx = torch.vstack(pertbx)

            # 不需要计算梯度，提高计算效率
            with torch.no_grad():
                # 将真实标签复制若干次以匹配所有的攻击列表
                red_label_ext = torch.cat([red_label] * len(self.attack_list))

                # 获取每种攻击方法产生的损失值和成功状态
                if not oblivion:
                    purified_pertbx = purifier(
                        pertbx.detach().clone().float()).to(torch.double)
                else:
                    purified_pertbx = pertbx.detach().clone()

                loss, done = self.get_scores(
                    predict_model, purified_pertbx, red_label_ext)

                # 调整损失和成功状态的形状以方便后续计算
                loss = loss.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)

                # 判断哪些样本至少有一种攻击方法成功
                success_flag = torch.any(done, dim=-1)

                # 对于没有成功的样本，将其标记为1以进行后续处理
                done[~torch.any(done, dim=-1)] = 1

                # 调整损失值，对于成功的攻击方法，损失值保持不变；对于失败的，损失值变为最小值
                loss = (loss * done.to(torch.float)) + \
                    torch.min(loss) * (~done).to(torch.float)

                # 调整扰动数据的形状以方便后续计算
                pertbx = pertbx.reshape(
                    len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])

                # 选择造成最大损失的扰动数据
                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(
                    num_sample_red), indices]

                # 获取选中的扰动数据的损失值
                a_loss = loss[torch.arange(num_sample_red), indices]

                # 复制当前的停止标志
                pre_stop_flag = stop_flag.clone()

                # 更新停止标志，如果损失值变化很小或者某种攻击方法成功，则停止迭代
                stop_flag[~stop_flag] = (
                    torch.abs(pre_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag

                # 更新前一个损失值
                pre_loss[~pre_stop_flag] = a_loss

            # 如果需要打印日志
            if verbose:
                # 评估最终的扰动数据的成功状态
                with torch.no_grad():
                    purified_adv_x = purifier(
                        adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_scores(
                        predict_model, purified_adv_x, label)
                    # 打印攻击成功率
                    logger.info(
                        f"max: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

            # 返回最终的扰动数据
            return adv_x

    # 这个get_scores函数的主要目的是计算扰动数据在给定模型上的损失值，并判断模型对这些扰动数据的预测是否成功完成。
    # 对于具有检测器功能的模型，还会考虑模型的额外输出来决定预测的完成状态。

    def get_scores(self, model, pertb_x, label):
        """
        获取扰动数据在模型上的损失值和预测标签的完成状态。

        参数：
        @param model: 模型对象，即受攻击的目标模型。
        @param pertb_x: torch.Tensor，扰动后的数据。
        @param label: torch.Tensor，扰动数据的真实标签。

        返回：
        - loss_no_reduction: 每个样本的损失值（无降维处理）。
        - done: Boolean Tensor，表示模型对每个样本的预测是否成功完成。
        """
        # 判断模型是否具有检测器功能，如果有，则获取模型的两个输出：logits_f 和 prob_g。
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(pertb_x)
        else:
            # 如果模型没有检测器功能，只获取一个输出logits_f。
            logits_f = model.forward(pertb_x)

        # 使用交叉熵计算每个样本的损失值
        ce = F.cross_entropy(logits_f, label, reduction='none')

        # 获取模型的预测标签
        y_pred = logits_f.argmax(1)

        # 如果模型具有检测器功能且不处于"oblivion"模式，则进行特殊处理。
        # 使用模型的输出prob_g来判断是否成功完成了预测。
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)
            loss_no_reduction = -prob_g
            done = (y_pred != label) & (prob_g <= tau)
        else:
            # 如果模型没有检测器功能或处于"oblivion"模式，则使用交叉熵损失来判断是否成功完成了预测。
            loss_no_reduction = ce
            done = y_pred != label

        return loss_no_reduction, done
