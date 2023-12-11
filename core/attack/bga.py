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
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, xor_tensors, or_tensors
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.bga')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BGA(BaseAttack):
    """
    Multi-step bit gradient ascent

    Parameters
    ---------
    @param is_attacker, Boolean, if ture means the role is the attacker
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence on adversary indicator
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BGA, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.omega = None           # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are permitted to be insertable
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None,
             m=10,
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
        m : int, 默认为10
            扰动的最大次数，与论文中的hp k相对应。
        lmda : float, 默认为1.0
            平衡对手检测器重要性的惩罚因子。
        use_sample : bool, 默认为False
            是否使用随机的起始点。
        
        返回:
        ----------
        worst_x : torch.FloatTensor
            经过扰动后的特征向量。
        """

        # 检查输入x是否为空或无效
        if x is None or x.shape[0] <= 0:
            return []

        # 初始化
        sqrt_m = torch.from_numpy(np.sqrt([x.size()[1]])).float().to(model.device)
        adv_x = x.clone()
        worst_x = x.detach().clone()

        # 将模型设置为评估模式
        model.eval()

        # 获取扰动的起始点
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)

        # 进行m次扰动
        for t in range(m):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmda)
            
            # 保存有效的扰动结果
            worst_x[done] = adv_x[done]
            
            # 计算梯度
            grads = torch.autograd.grad(loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            # 计算更新
            x_update = (sqrt_m * (1. - 2. * adv_x) * grad >= torch.norm(
                grad, 2, 1).unsqueeze(1).expand_as(adv_x)).float()
            
            # 对特征向量进行扰动
            adv_x = xor_tensors(x_update, adv_x)
            adv_x = or_tensors(adv_x, x)

            # 选择对抗样本
            done = self.get_scores(model, adv_x, label).data
            worst_x[done] = adv_x[done]

        return worst_x


    def perturb(self, model, x, label=None,
                steps=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                use_sample=False,
                base=10.,
                verbose=False):
        """
        对输入数据x进行扰动，使其在给定模型上的输出发生改变。
        
        参数:
        ----------
        model : PyTorch模型
            需要进行攻击的目标模型。
        x : torch.Tensor
            输入数据。
        label : torch.Tensor, 可选
            输入数据的真实标签。
        steps : int, 默认为10
            攻击步数。
        min_lambda_ : float, 默认为1e-5
            攻击强度的最小值。
        max_lambda_ : float, 默认为1e5
            攻击强度的最大值。
        use_sample : bool, 默认为False
            是否使用样本。
        base : float, 默认为10.0
            用于调整lambda的基数。
        verbose : bool, 默认为False
            是否打印详细信息。
            
        返回:
        ----------
        adv_x : torch.Tensor
            经过扰动的输入数据。
        """

        # 确保lambda的范围是合法的
        assert 0 < min_lambda_ <= max_lambda_

        # 将模型设置为评估模式
        model.eval()

        # 根据模型属性设置lambda的初始值
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # 创建一个与x相同的可扰动版本
        adv_x = x.detach().clone().to(torch.double)

        # 在指定的lambda范围内进行扰动
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if torch.all(done):
                break
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                steps,
                                lmda=self.lambda_,
                                use_sample=use_sample
                                )
            adv_x[~done] = pert_x
            self.lambda_ *= base

        # 获取最终的损失值
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)

            # 如果设置了详细输出，打印攻击效果
            if verbose:
                logger.info(f"BGA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x
