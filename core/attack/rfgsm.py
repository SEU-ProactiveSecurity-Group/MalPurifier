"""
@inproceedings{al2018adversarial,
  title={Adversarial deep learning for robust detection of binary encoded malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O’Reilly, Una-May},
  booktitle={2018 IEEE Security and Privacy Workshops (SPW)},
  pages={76--82},
  year={2018},
  organization={IEEE}
}
"""

import torch

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x, or_tensors
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.rfgsm')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class RFGSM(BaseAttack):
    """
    FGSM^k with randomized rounding

    Parameters
    ---------
    @param is_attacker, Boolean, if ture means the role is the attacker
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence on adversary indicator
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, random=False, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(RFGSM, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are permitted to be insertable
        self.lmba = 1.
        self.random = random

    def _perturb(self, model, x, label=None,
                steps=10,
                step_length=0.02,
                lmda=1.,
                use_sample=False):
        """
        扰动节点的特征向量。

        参数:
        ----------
        model : PyTorch模型
            被攻击的目标模型。
        x : torch.FloatTensor
            具有形状[batch_size, vocab_dim]的特征向量。
        label : torch.LongTensor, 可选
            数据的真实标签。
        steps : int, 默认为10
            最大的迭代次数。
        step_length : float, 默认为0.02
            每个方向上的更新值。
        lmda : float, 默认为1.0
            平衡对手检测器重要性的惩罚因子。
        use_sample : bool, 默认为False
            是否使用随机的起始点。

        返回:
        ----------
        adv_x : torch.FloatTensor
            经过扰动后的特征向量。
        """
        
        # 检查输入x是否为空或无效
        if x is None or x.shape[0] <= 0:
            return []

        adv_x = x.clone()
        
        # 将模型设置为评估模式
        model.eval()

        # 获取扰动的起始点
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)
        loss_natural = 0.

        # 进行steps次迭代，每次迭代都根据当前梯度对adv_x进行更新
        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmda)
            
            # 保存第一次迭代的损失作为自然损失
            if t == 0:
                loss_natural = loss
            
            # 计算梯度
            grads = torch.autograd.grad(loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            # 过滤不需要的图形和位置
            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)
            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)

            # 根据梯度更新adv_x
            adv_x = torch.clamp(adv_x + step_length * torch.sign(grad4ins_), min=0., max=1.)

        # 如果self.random为True，生成一个随机阈值，否则使用0.5作为阈值
        if self.random:
            round_threshold = torch.rand(adv_x.size()).to(self.device)
        else:
            round_threshold = 0.5

        # 使用阈值对adv_x进行舍入
        adv_x = round_x(adv_x, round_threshold)

        # 执行可行性投影
        adv_x = or_tensors(adv_x, x)

        # 与官方代码不同，因为设计一个合适的评分测量是具有挑战性的
        loss_adv, _1 = self.get_loss(model, adv_x, label, lmda)

        # 如果对抗损失小于自然损失，则用原始x替换adv_x中的对应位置
        replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(adv_x)
        adv_x[replace_flag] = x[replace_flag]

        return adv_x


    def perturb(self, model, x, label=None,
                steps=10,
                step_length=0.02,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False,
                use_sample=False):
        """
        对输入数据进行增强攻击。

        参数:
        ----------
        model : PyTorch模型
            被攻击的目标模型。
        x : torch.FloatTensor
            具有形状[batch_size, vocab_dim]的特征向量。
        label : torch.LongTensor, 可选
            数据的真实标签。
        steps : int, 默认为10
            _perturb方法中的最大迭代次数。
        step_length : float, 默认为0.02
            每个方向上的更新值。
        min_lambda_ : float, 默认为1e-5
            lambda的最小值。
        max_lambda_ : float, 默认为1e5
            lambda的最大值。
        base : float, 默认为10.0
            lambda的更新因子。
        verbose : bool, 默认为False
            是否打印详细日志。
        use_sample : bool, 默认为False
            _perturb方法中是否使用随机的起始点。

        返回:
        ----------
        adv_x : torch.FloatTensor
            经过扰动后的特征向量。
        """

        # 确保lambda的值在一个合理的范围内
        assert 0 < min_lambda_ <= max_lambda_

        # 将模型设置为评估模式
        model.eval()

        # 检查模型是否有一个名为'is_detector_enabled'的属性，并据此设置lambda的值
        if hasattr(model, 'is_detector_enabled'):
            self.lmba = min_lambda_
        else:
            self.lmba = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        # 当lambda的值小于或等于其最大值时，循环执行以下操作
        while self.lmba <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lmba)
            # 如果所有样本都达到了目标，跳出循环
            if torch.all(done):
                break
            # 对那些还未达到目标的样本进行扰动
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                steps,
                                step_length,
                                lmda=self.lmba,
                                use_sample=use_sample
                                )
            adv_x[~done] = pert_x
            # 更新lambda的值
            self.lmba *= base

        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lmba)
        # 如果设置了详细模式，打印攻击的有效性
        if verbose:
            logger.info(f"rFGSM: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x


    def perturb_dae(self, model, purifier, x, label=None,
                    steps=10,
                    step_length=0.02,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    base=10.,
                    verbose=False,
                    use_sample=False,
                    oblivion=False):

            # 确保lambda的值在一个合理的范围内
            assert 0 < min_lambda_ <= max_lambda_

            # 将模型设置为评估模式
            model.eval()

            # 检查模型是否有一个名为'is_detector_enabled'的属性，并据此设置lambda的值
            if hasattr(model, 'is_detector_enabled'):
                self.lmba = min_lambda_
            else:
                self.lmba = max_lambda_

            adv_x = x.detach().clone().to(torch.double)

            # 当lambda的值小于或等于其最大值时，循环执行以下操作
            while self.lmba <= max_lambda_:
                with torch.no_grad():
                    if not oblivion:
                        purified_adv = purifier(adv_x.detach().clone().float()).to(torch.double)
                        _, done = self.get_loss(model, purified_adv, label, self.lmba)
                    else:
                        _, done = self.get_loss(model, adv_x, label, self.lmba)
                # 如果所有样本都达到了目标，跳出循环
                if torch.all(done):
                    break
                # 对那些还未达到目标的样本进行扰动
                pert_x = self._perturb(model, adv_x[~done], label[~done],
                                    steps,
                                    step_length,
                                    lmda=self.lmba,
                                    use_sample=use_sample
                                    )
                adv_x[~done] = pert_x
                # 更新lambda的值
                self.lmba *= base

            with torch.no_grad():
                purified_adv = purifier(adv_x.detach().clone().float()).to(torch.double)
                _, done = self.get_loss(model, purified_adv, label, self.lmba)
            # 如果设置了详细模式，打印攻击的有效性
            if verbose:
                logger.info(f"rFGSM: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

            return adv_x