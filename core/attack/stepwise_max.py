import torch
import torch.nn.functional as F

import random
from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.stepwise_max')
logger.addHandler(ErrorHandler)
EXP_OVER_FLOW = 1e-120


class StepwiseMax(BaseAttack):
    """
    Stepwise max攻击方法，这是一个结合了pgd l1, pgd l2, 和 pgd linf三种攻击方式的方法。

    参数
    ----------
    @param use_random: bool类型，是否使用随机的起始点。
    @param rounding_threshold: float类型，用于四舍五入实数的阈值。
    @param is_attacker: bool类型，是否扮演攻击者角色（注意：防御者执行对抗性训练）。
    @param oblivion: bool类型，是否知道敌手指示器。
    @param kappa: 攻击信心度。
    @param manipulation_x: 可操作性。
    @param omega: 与每个api相对应的互依赖api的索引。
    @param device: 设备，'cpu'或'cuda'。

    """

    def __init__(self, use_random=False, rounding_threshold=0.5,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(StepwiseMax, self).__init__(is_attacker,
                                          oblivion, kappa, manipulation_x, omega, device)

        # 是否使用随机起点
        self.use_random = use_random

        # 断言确保四舍五入阈值在(0, 1)之间
        assert 0 < rounding_threshold < 1

        # 设置四舍五入的阈值
        self.round_threshold = rounding_threshold

        # lambda_用于正则化，通常与优化的损失一起使用
        self.lambda_ = 1.

    def perturb_dae(self, model, purifier, x, label=None,
                    steps=100,
                    step_check=1,
                    sl_l1=1.,
                    sl_l2=1.,
                    sl_linf=0.01,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    is_score_round=True,
                    base=10.,
                    verbose=False,
                    oblivion=False):
        """
        对模型进行增强攻击。

        @param model: PyTorch模型，待攻击目标。
        @param x: Tensor, 原始输入数据。
        @param label: Tensor或None, 输入数据对应的标签。
        @param steps: int, 攻击的总步数。
        @param step_check: int, 检查间隔，即多少步进行一次检查。
        @param sl_l1: float, L1范数的步长。
        @param sl_l2: float, L2范数的步长。
        @param sl_linf: float, Linf范数的步长。
        @param min_lambda_: float, lambda的最小值。
        @param max_lambda_: float, lambda的最大值。
        @param is_score_round: Boolean, 是否对分数进行四舍五入。
        @param base: float, 基数。
        @param verbose: Boolean, 是否输出详细信息。
        """
        # torch.manual_seed(int(random.random() * 100))  # 设置随机种子
        # 参数校验
        assert 0 < min_lambda_ <= max_lambda_
        assert steps >= 0 and (
            step_check >= 1) and 1 >= sl_l1 > 0 and sl_l2 >= 0 and sl_linf >= 0

        model.eval()  # 将模型设置为评估模式
        purifier.eval()

        # 根据模型是否具有某种属性来设置lambda的初值
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # 如果不是攻击者，从预定义的步骤中随机选择一个
        if not self.is_attacker:
            step_checks = [1, 10, 25, 50]
            step_check = random.choice(step_checks)

        # 计算每个小步骤中需要的迭代次数
        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + \
            [steps % step_check] if steps % step_check != 0 else mini_steps

        # 获取输入的维度信息
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        adv_x = x.detach().clone()  # 获取输入数据的副本
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            for i, mini_step in enumerate(mini_steps):
                with torch.no_grad():
                    # 如果是第一步并且启用了随机初始化，那么获取一个随机的起始点
                    if i == 0:
                        adv_x = get_x0(
                            adv_x, rounding_threshold=self.round_threshold, is_sample=True)
                    # 计算损失和完成标志
                    if not oblivion:
                        purified_adv = purifier(
                            adv_x.detach().clone().float()).to(torch.double)
                    else:
                        purified_adv = adv_x.detach().clone()
                    _, done = self.get_loss(
                        model, purified_adv, label, self.lambda_)

                # print("done:", done)

                # 如果所有的都完成了，就退出循环
                if torch.all(done):
                    break

                # 对于那些没有完成的数据，重新计算扰动
                # print("i:", i)
                if i == 0:
                    # print("~done:", (~done))
                    adv_x[~done] = x[~done]
                    prev_done = done.clone()
                else:
                    if (adv_x[~done]).shape[0] == (pert_x_cont[~done[~prev_done]]).shape[0]:
                        adv_x[~done] = pert_x_cont[~done[~prev_done]]
                    else:
                        updated_mask = (~done) & (~prev_done[:len(done)])
                        num_to_select = updated_mask.sum().item()
                        selected_perturbations = pert_x_cont[:num_to_select]
                        adv_x[updated_mask] = selected_perturbations

                prev_done = done.clone()

                # 对那些未完成的数据进行真正的扰动
                num_sample_red = torch.sum(~done).item()
                pert_x_l1, pert_x_l2, pert_x_linf = self._perturb_dae(model, purifier, adv_x[~done], label[~done],
                                                                      mini_step,
                                                                      sl_l1,
                                                                      sl_l2,
                                                                      sl_linf,
                                                                      lambda_=self.lambda_,
                                                                      oblivion=False
                                                                      )
                # print("pert_x_l1, pert_x_l2, pert_x_linf", pert_x_l1, pert_x_l2, pert_x_linf)
                # 不计算梯度地执行下列操作
                with torch.no_grad():
                    # 构造一个包含三种扰动的列表
                    pertb_x_list = [pert_x_linf, pert_x_l2, pert_x_l1]
                    n_attacks = len(pertb_x_list)  # 获取攻击的数量（即3）
                    pertbx = torch.vstack(pertb_x_list)  # 垂直堆叠这三种扰动
                    # 扩展标签列表，使其与扰动列表长度匹配
                    label_ext = torch.cat([label[~done]] * n_attacks)

                    # 如果不是攻击者并且不需要四舍五入得分，则获取得分
                    # 否则，先对扰动进行四舍五入，再获取得分
                    if not oblivion:
                        purified_pertbx = purifier(
                            pertbx.detach().clone().float()).to(torch.double)
                    else:
                        purified_pertbx = pertbx.detach().clone()
                    if (not self.is_attacker) and (not is_score_round):
                        scores, _done = self.get_scores(
                            model, purified_pertbx, label_ext)
                    else:
                        scores, _done = self.get_scores(model, round_x(
                            purified_pertbx, self.round_threshold), label_ext)

                    # 如果得分的最大值大于0，则设置为该值，否则设置为0
                    max_v = scores.amax() if scores.amax() > 0 else 0.
                    scores[_done] += max_v  # 对完成的得分增加max_v

                    # 重新整形扰动和得分张量，以便后续操作
                    pertbx = pertbx.reshape(
                        n_attacks, num_sample_red, *red_n).permute([1, 0, *red_ind])
                    scores = scores.reshape(
                        n_attacks, num_sample_red).permute(1, 0)

                    # 从得分张量中获取最大得分及其索引
                    _2, s_idx = scores.max(dim=-1)
                    # 使用索引从扰动张量中选择具有最高误导性的扰动
                    pert_x_cont = pertbx[torch.arange(num_sample_red), s_idx]
                    # print("pert_x_cont.shape", pert_x_cont.shape)
                    # 更新经过扰动的数据adv_x
                    adv_x[~done] = pert_x_cont if not self.is_attacker else round_x(
                        pert_x_cont, self.round_threshold)

            # 更新lambda值以便于下一次循环
            self.lambda_ *= base
            # 如果lambda值检查失败，则中断循环
            if not self.check_lambda(model):
                break
        # 如果是攻击者，对最终的扰动结果进行四舍五入
        if self.is_attacker:
            adv_x = round_x(adv_x, self.round_threshold)

        # 不计算梯度地获取最后的损失和完成标志
        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lambda_)
            # 如果设置了详细输出，打印攻击效果的百分比
            if verbose:
                logger.info(
                    f"step-wise max: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        # 返回扰动后的数据
        return adv_x

    def perturb(self, model, x, label=None,
                steps=100,
                step_check=1,
                sl_l1=1.,
                sl_l2=1.,
                sl_linf=0.01,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                is_score_round=True,
                base=10.,
                verbose=False):
        """
        对模型进行增强攻击。

        @param model: PyTorch模型，待攻击目标。
        @param x: Tensor, 原始输入数据。
        @param label: Tensor或None, 输入数据对应的标签。
        @param steps: int, 攻击的总步数。
        @param step_check: int, 检查间隔，即多少步进行一次检查。
        @param sl_l1: float, L1范数的步长。
        @param sl_l2: float, L2范数的步长。
        @param sl_linf: float, Linf范数的步长。
        @param min_lambda_: float, lambda的最小值。
        @param max_lambda_: float, lambda的最大值。
        @param is_score_round: Boolean, 是否对分数进行四舍五入。
        @param base: float, 基数。
        @param verbose: Boolean, 是否输出详细信息。
        """
        # torch.manual_seed(int(random.random() * 100))  # 设置随机种子
        # 参数校验
        assert 0 < min_lambda_ <= max_lambda_
        assert steps >= 0 and (
            step_check >= 1) and 1 >= sl_l1 > 0 and sl_l2 >= 0 and sl_linf >= 0

        model.eval()  # 将模型设置为评估模式

        # 根据模型是否具有某种属性来设置lambda的初值
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # 如果不是攻击者，从预定义的步骤中随机选择一个
        if not self.is_attacker:
            step_checks = [1, 10, 25, 50]
            step_check = random.choice(step_checks)

        # 计算每个小步骤中需要的迭代次数
        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + \
            [steps % step_check] if steps % step_check != 0 else mini_steps

        # 获取输入的维度信息
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        adv_x = x.detach().clone()  # 获取输入数据的副本
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            for i, mini_step in enumerate(mini_steps):
                with torch.no_grad():
                    # 如果是第一步并且启用了随机初始化，那么获取一个随机的起始点
                    if i == 0:
                        adv_x = get_x0(
                            adv_x, rounding_threshold=self.round_threshold, is_sample=True)
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)

                # print("done:", done)

                # 如果所有的都完成了，就退出循环
                if torch.all(done):
                    break

                # 对于那些没有完成的数据，重新计算扰动
                # print("i:", i)
                if i == 0:
                    # print("~done:", (~done))
                    adv_x[~done] = x[~done]
                    prev_done = done.clone()
                else:
                    if (adv_x[~done]).shape[0] == (pert_x_cont[~done[~prev_done]]).shape[0]:
                        adv_x[~done] = pert_x_cont[~done[~prev_done]]
                    else:
                        updated_mask = (~done) & (~prev_done[:len(done)])
                        num_to_select = updated_mask.sum().item()
                        selected_perturbations = pert_x_cont[:num_to_select]
                        adv_x[updated_mask] = selected_perturbations

                prev_done = done.clone()

                # 对那些未完成的数据进行真正的扰动
                num_sample_red = torch.sum(~done).item()
                pert_x_l1, pert_x_l2, pert_x_linf = self._perturb(model, adv_x[~done], label[~done],
                                                                  mini_step,
                                                                  sl_l1,
                                                                  sl_l2,
                                                                  sl_linf,
                                                                  lambda_=self.lambda_
                                                                  )
                # print("pert_x_l1, pert_x_l2, pert_x_linf", pert_x_l1, pert_x_l2, pert_x_linf)
                # 不计算梯度地执行下列操作
                with torch.no_grad():
                    # 构造一个包含三种扰动的列表
                    pertb_x_list = [pert_x_linf, pert_x_l2, pert_x_l1]
                    n_attacks = len(pertb_x_list)  # 获取攻击的数量（即3）
                    pertbx = torch.vstack(pertb_x_list)  # 垂直堆叠这三种扰动
                    # 扩展标签列表，使其与扰动列表长度匹配
                    label_ext = torch.cat([label[~done]] * n_attacks)

                    # 如果不是攻击者并且不需要四舍五入得分，则获取得分
                    # 否则，先对扰动进行四舍五入，再获取得分
                    if (not self.is_attacker) and (not is_score_round):
                        scores, _done = self.get_scores(
                            model, pertbx, label_ext)
                    else:
                        scores, _done = self.get_scores(model, round_x(
                            pertbx, self.round_threshold), label_ext)

                    # 如果得分的最大值大于0，则设置为该值，否则设置为0
                    max_v = scores.amax() if scores.amax() > 0 else 0.
                    scores[_done] += max_v  # 对完成的得分增加max_v

                    # 重新整形扰动和得分张量，以便后续操作
                    pertbx = pertbx.reshape(
                        n_attacks, num_sample_red, *red_n).permute([1, 0, *red_ind])
                    scores = scores.reshape(
                        n_attacks, num_sample_red).permute(1, 0)

                    # 从得分张量中获取最大得分及其索引
                    _2, s_idx = scores.max(dim=-1)
                    # 使用索引从扰动张量中选择具有最高误导性的扰动
                    pert_x_cont = pertbx[torch.arange(num_sample_red), s_idx]
                    # print("pert_x_cont.shape", pert_x_cont.shape)
                    # 更新经过扰动的数据adv_x
                    adv_x[~done] = pert_x_cont if not self.is_attacker else round_x(
                        pert_x_cont, self.round_threshold)

            # 更新lambda值以便于下一次循环
            self.lambda_ *= base
            # 如果lambda值检查失败，则中断循环
            if not self.check_lambda(model):
                break
        # 如果是攻击者，对最终的扰动结果进行四舍五入
        if self.is_attacker:
            adv_x = round_x(adv_x, self.round_threshold)

        # 不计算梯度地获取最后的损失和完成标志
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            # 如果设置了详细输出，打印攻击效果的百分比
            if verbose:
                logger.info(
                    f"step-wise max: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        # 返回扰动后的数据
        return adv_x

    def _perturb(self, model, x, label=None,
                 steps=1,
                 step_length_l1=1.,
                 step_length_l2=0.5,
                 step_length_linf=0.01,
                 lambda_=1.,
                 ):
        """
        对节点的特征向量进行扰动

        参数
        -----------
        @param model: 受害者模型
        @param x: torch.FloatTensor, 节点特征向量（每个表示一个图中的API出现次数）形状为 [batch_size, vocab_dim]
        @param label: torch.LongTensor, 真实的标签
        @param steps: 整数, 迭代的最大次数
        @param step_length_l1: 每次迭代的步长，L1范数
        @param step_length_l2: 每次迭代的步长，L2范数
        @param step_length_linf: 每次迭代的步长，Linf范数
        @param lambda_: 浮点数, 惩罚因子
        """
        if x is None or x.shape[0] <= 0:
            return []

        self.lambda_ = lambda_

        # 确保L1步长在[0,1]之间
        assert 0 <= step_length_l1 <= 1, "期望在 [0,1] 之间的实数值，但得到 {}".format(
            step_length_l1)
        model.eval()
        adv_x = x.detach()

        def one_iteration(_adv_x, norm_type):
            # 基于当前的扰动输入来计算梯度
            if "rnn" in model.model_save_path:
                model.train()
            if "lstm" in model.model_save_path:
                model.train()
            var_adv_x = torch.autograd.Variable(
                _adv_x, requires_grad=True)  # 将_adv_x转换为一个可以进行自动梯度计算的变量
            loss, done = self.get_loss(
                model, var_adv_x, label, self.lambda_)  # 获取模型在扰动输入上的损失
            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            if grads[0] is None:
                grad = torch.zeros_like(var_adv_x)
            else:
                grad = grads[0].data

            # 寻找允许的位置来插入和移除API
            pos_insertion = (_adv_x <= 0.5) * 1 * \
                (_adv_x >= 0.)  # 寻找API的可插入位置：特征值在0和0.5之间
            grad4insertion = (grad > 0) * pos_insertion * \
                grad  # 根据梯度正值计算插入API的梯度

            pos_removal = (_adv_x > 0.5) * 1  # 寻找API的可移除位置：特征值大于0.5
            grad4removal = (grad <= 0) * (pos_removal &
                                          self.manipulation_x) * grad  # 根据梯度负值计算移除API的梯度

            if self.is_attacker:
                # 对于攻击者，处理那些互相依赖的API
                checking_nonexist_api = (
                    pos_removal ^ self.omega) & self.omega  # 检查不存在的API
                # 考虑API之间的关系，调整移除API的梯度
                grad4removal[:, self.api_flag] += torch.sum(
                    grad * checking_nonexist_api, dim=-1, keepdim=True)

            # 合并插入和移除的梯度
            grad = grad4removal + grad4insertion

            # 根据不同的范数类型，计算扰动值
            if norm_type == 'linf':
                perturbation = torch.sign(grad)  # 计算梯度符号来获取无穷范数扰动方向
                if self.is_attacker:
                    perturbation += (torch.any(
                        perturbation[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
                # 应用扰动并确保结果在[0,1]范围内
                return torch.clamp(_adv_x + step_length_linf * perturbation, min=0., max=1.)

            elif norm_type == 'l2':
                l2norm = torch.linalg.norm(
                    grad, dim=-1, keepdim=True)  # 计算L2范数
                perturbation = torch.minimum(
                    torch.tensor(1., dtype=_adv_x.dtype, device=_adv_x.device),
                    grad / l2norm
                )  # 计算L2范数下的扰动方向
                perturbation = torch.where(torch.isnan(
                    perturbation), 0., perturbation)  # 处理NaN值
                perturbation = torch.where(torch.isinf(
                    perturbation), 1., perturbation)  # 处理Inf值
                if self.is_attacker:
                    min_val = torch.amin(
                        perturbation, dim=-1, keepdim=True).clamp_(max=0.)
                    perturbation += (torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                                     keepdim=True) * torch.abs(min_val) * checking_nonexist_api)
                return torch.clamp(_adv_x + step_length_l2 * perturbation, min=0., max=1.)

            elif norm_type == 'l1':
                val, idx = torch.abs(grad).topk(
                    int(1. / step_length_l1), dim=-1)  # 获取梯度的绝对值的top-k值和相应的索引
                perturbation = F.one_hot(
                    idx, num_classes=_adv_x.shape[-1]).sum(dim=1)  # 根据索引计算L1范数下的扰动方向
                perturbation = torch.sign(
                    grad) * perturbation  # 使用梯度的符号来调整扰动方向
                if self.is_attacker:
                    perturbation += (torch.any(
                        perturbation[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
                return torch.clamp(_adv_x + step_length_l1 * perturbation, min=0., max=1.)

            else:
                raise NotImplementedError  # 如果范数类型不在L1、L2、Linf中，则引发异常

        # 为每种范数执行迭代
        adv_x_l1 = adv_x.clone()
        for t in range(steps):
            adv_x_l1 = one_iteration(adv_x_l1, norm_type='l1')

        adv_x_l2 = adv_x.clone()
        for t in range(steps):
            adv_x_l2 = one_iteration(adv_x_l2, norm_type='l2')

        adv_x_linf = adv_x.clone()
        for t in range(steps):
            adv_x_linf = one_iteration(adv_x_linf, norm_type='linf')

        return adv_x_l1, adv_x_l2, adv_x_linf

    def _perturb_dae(self, model, purifier, x, label=None,
                     steps=1,
                     step_length_l1=1.,
                     step_length_l2=0.5,
                     step_length_linf=0.01,
                     lambda_=1.,
                     oblivion=False):
        """
        对节点的特征向量进行扰动

        参数
        -----------
        @param model: 受害者模型
        @param x: torch.FloatTensor, 节点特征向量（每个表示一个图中的API出现次数）形状为 [batch_size, vocab_dim]
        @param label: torch.LongTensor, 真实的标签
        @param steps: 整数, 迭代的最大次数
        @param step_length_l1: 每次迭代的步长，L1范数
        @param step_length_l2: 每次迭代的步长，L2范数
        @param step_length_linf: 每次迭代的步长，Linf范数
        @param lambda_: 浮点数, 惩罚因子
        """
        if x is None or x.shape[0] <= 0:
            return []

        self.lambda_ = lambda_

        # 确保L1步长在[0,1]之间
        assert 0 <= step_length_l1 <= 1, "期望在 [0,1] 之间的实数值，但得到 {}".format(
            step_length_l1)
        model.eval()
        adv_x = x.detach()

        def one_iteration(_adv_x, norm_type):
            # 基于当前的扰动输入来计算梯度
            var_adv_x = torch.autograd.Variable(
                _adv_x, requires_grad=True)  # 将_adv_x转换为一个可以进行自动梯度计算的变量
            if not oblivion:
                purified_var = purifier(
                    var_adv_x.detach().clone().float()).to(torch.double)
            else:
                purified_var = var_adv_x.detach().clone()
            loss, done = self.get_loss(
                model, purified_var, label, self.lambda_)  # 获取模型在扰动输入上的损失
            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            if grads[0] is None:
                grad = torch.zeros_like(var_adv_x)
            else:
                grad = grads[0].data

            # 寻找允许的位置来插入和移除API
            pos_insertion = (_adv_x <= 0.5) * 1 * \
                (_adv_x >= 0.)  # 寻找API的可插入位置：特征值在0和0.5之间
            grad4insertion = (grad > 0) * pos_insertion * \
                grad  # 根据梯度正值计算插入API的梯度

            pos_removal = (_adv_x > 0.5) * 1  # 寻找API的可移除位置：特征值大于0.5
            grad4removal = (grad <= 0) * (pos_removal &
                                          self.manipulation_x) * grad  # 根据梯度负值计算移除API的梯度

            if self.is_attacker:
                # 对于攻击者，处理那些互相依赖的API
                checking_nonexist_api = (
                    pos_removal ^ self.omega) & self.omega  # 检查不存在的API
                # 考虑API之间的关系，调整移除API的梯度
                grad4removal[:, self.api_flag] += torch.sum(
                    grad * checking_nonexist_api, dim=-1, keepdim=True)

            # 合并插入和移除的梯度
            grad = grad4removal + grad4insertion

            # 根据不同的范数类型，计算扰动值
            if norm_type == 'linf':
                perturbation = torch.sign(grad)  # 计算梯度符号来获取无穷范数扰动方向
                if self.is_attacker:
                    perturbation += (torch.any(
                        perturbation[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
                # 应用扰动并确保结果在[0,1]范围内
                return torch.clamp(_adv_x + step_length_linf * perturbation, min=0., max=1.)

            elif norm_type == 'l2':
                l2norm = torch.linalg.norm(
                    grad, dim=-1, keepdim=True)  # 计算L2范数
                perturbation = torch.minimum(
                    torch.tensor(1., dtype=_adv_x.dtype, device=_adv_x.device),
                    grad / l2norm
                )  # 计算L2范数下的扰动方向
                perturbation = torch.where(torch.isnan(
                    perturbation), 0., perturbation)  # 处理NaN值
                perturbation = torch.where(torch.isinf(
                    perturbation), 1., perturbation)  # 处理Inf值
                if self.is_attacker:
                    min_val = torch.amin(
                        perturbation, dim=-1, keepdim=True).clamp_(max=0.)
                    perturbation += (torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                                     keepdim=True) * torch.abs(min_val) * checking_nonexist_api)
                return torch.clamp(_adv_x + step_length_l2 * perturbation, min=0., max=1.)

            elif norm_type == 'l1':
                val, idx = torch.abs(grad).topk(
                    int(1. / step_length_l1), dim=-1)  # 获取梯度的绝对值的top-k值和相应的索引
                perturbation = F.one_hot(
                    idx, num_classes=_adv_x.shape[-1]).sum(dim=1)  # 根据索引计算L1范数下的扰动方向
                perturbation = torch.sign(
                    grad) * perturbation  # 使用梯度的符号来调整扰动方向
                if self.is_attacker:
                    perturbation += (torch.any(
                        perturbation[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
                return torch.clamp(_adv_x + step_length_l1 * perturbation, min=0., max=1.)

            else:
                raise NotImplementedError  # 如果范数类型不在L1、L2、Linf中，则引发异常

        # 为每种范数执行迭代
        adv_x_l1 = adv_x.clone()
        for t in range(steps):
            adv_x_l1 = one_iteration(adv_x_l1, norm_type='l1')

        adv_x_l2 = adv_x.clone()
        for t in range(steps):
            adv_x_l2 = one_iteration(adv_x_l2, norm_type='l2')

        adv_x_linf = adv_x.clone()
        for t in range(steps):
            adv_x_linf = one_iteration(adv_x_linf, norm_type='linf')

        return adv_x_l1, adv_x_l2, adv_x_linf

    def get_scores(self, model, pertb_x, label):
        # 如果模型有 'is_detector_enabled' 这个属性
        if hasattr(model, 'is_detector_enabled'):
            # 获取模型的输出，logits_f 是模型的原始输出，prob_g 是一个概率值
            logits_f, prob_g = model.forward(pertb_x)
        else:
            # 如果模型没有 'is_detector_enabled' 这个属性，只获取模型的原始输出
            logits_f = model.forward(pertb_x)

        # 获取预测的类别
        y_pred = logits_f.argmax(1)

        # 计算交叉熵损失
        ce = F.cross_entropy(logits_f, label, reduction='none')

        # 如果模型有 'is_detector_enabled' 这个属性，并且 self.oblivion 为 False
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            # 获取样本的阈值
            tau = model.get_tau_sample_wise(y_pred)
            # 计算损失，加入了 prob_g 这个概率值的惩罚项
            loss_no_reduction = ce - self.lambda_ * prob_g
            # 判断预测是否错误，并且 prob_g 是否小于等于阈值 tau
            done = (y_pred != label) & (prob_g <= tau)
        else:
            # 如果没有 'is_detector_enabled' 这个属性或 self.oblivion 为 True，损失仍然是交叉熵损失
            loss_no_reduction = ce
            # 判断预测是否错误
            done = y_pred != label

        # 返回损失值和判断结果c
        return loss_no_reduction, done
