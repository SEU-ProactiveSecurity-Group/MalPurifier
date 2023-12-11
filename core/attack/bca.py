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

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.bca')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BCA(BaseAttack):
    """
    Multi-step bit coordinate ascent

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
        super(BCA, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.omega = None           # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are permitted to be insertable
        self.lambda_ = 1.


    def _perturb(self, model, x, label=None,
                steps=10, lmda=1., use_sample=False):
        """
        扰动节点特征向量。

        参数:
        -----------
        model : PyTorch模型
            受害模型。
        x : torch.FloatTensor
            形状为[batch_size, vocab_dim]的特征向量。
        label : torch.LongTensor, optional
            真实标签。
        steps : int, default=10
            最大扰动次数，即论文中的超参数k。
        lmda : float, default=1.
            用于平衡对手检测器重要性的惩罚因子。
        use_sample : bool, default=False
            是否使用随机起始点。
        
        返回:
        -------
        torch.FloatTensor
            最糟糕的对抗样本特征。
        """

        if x is None or x.shape[0] <= 0:
            return []  # 如果x为空或者x的shape[0]小于等于0，则返回空列表

        adv_x = x   # 对抗样本初始化为原始样本
        worst_x = x.detach().clone()  # 复制x作为"最糟糕的"对抗样本
        model.eval()  # 设置模型为评估模式
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)  # 获取扰动的起始点

        if "rnn" in model.model_save_path:
            model.train()
        if "lstm" in model.model_save_path:
            model.train()

        for t in range(steps):  # 进行指定次数的扰动
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)  # 创建一个需要求梯度的变量
            loss, done = self.get_loss(model, var_adv_x, label, lmda)  # 计算损失和完成标志

            # print("loss, done", loss, done)
            
            worst_x[done] = adv_x[done]  # 对于已经达到目的的样本，更新其为"最糟糕的"样本

            if torch.all(done):  # 所有样本都达到目的时，停止扰动
                break
            # print("var_adv_x", var_adv_x)
            grads = torch.autograd.grad(loss.mean(), var_adv_x)
            grad = grads[0].data

            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)         # 计算插入梯度的策略

            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            _2, pos = torch.max(grad4ins_, dim=-1)                      # 获取最大梯度的位置

            perturbation = F.one_hot(pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)  # 对此位置进行扰动

            perturbation[done] = 0.                                    # 对于已经达到目的的样本，不进行进一步扰动
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)  # 确保对抗样本的特征值在[0,1]之间

        done = self.get_scores(model, adv_x, label)
        worst_x[done] = adv_x[done]

        return worst_x  # 返回最糟糕的对抗样本
    
    def perturb(self, model, x, label=None,
                steps=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                use_sample=False,
                base=10.,
                verbose=False):
        """
        增强攻击
        """
        # 检查lambda_值的范围是否有效
        assert 0 < min_lambda_ <= max_lambda_
        
        # 将模型设为评估模式，这意味着模型中的所有dropout和batchnorm层将工作在评估模式  
        model.eval() 
        
        # 如果模型有属性is_detector_enabled
        if hasattr(model, 'is_detector_enabled'):  
            self.lambda_ = min_lambda_  # 则将lambda_设置为min_lambda_
        else:
            self.lambda_ = max_lambda_  # 否则将lambda_设置为max_lambda_
            
        # 创建输入x的副本，用于存储对抗样本的数据
        adv_x = x.detach().clone().to(torch.double) 
        
        # 当lambda_值小于或等于max_lambda_时，继续进行循环
        while self.lambda_ <= max_lambda_:  
            with torch.no_grad():  # 不需要计算梯度
                _, done = self.get_loss(model, adv_x, label, self.lambda_)  # 计算模型的损失和完成标志
                
            if torch.all(done):  # 如果所有的样本都已完成，即完成标志都为True
                break  # 则跳出循环
            
            pert_x = self._perturb(model, adv_x[~done], label[~done],  # 否则，对未完成的样本进行扰动
                                   steps,
                                   lmda=self.lambda_,
                                   use_sample=use_sample
                                   )
            
            adv_x[~done] = pert_x  # 更新对抗样本的数据
            self.lambda_ *= base  # 调整lambda_的值
            
        with torch.no_grad():  # 不需要计算梯度
            _, done = self.get_loss(model, adv_x, label, self.lambda_)  # 计算模型的损失和完成标志
            
            if verbose:  # 如果需要输出详细信息
                # 它会打印出攻击的有效性，即成功生成能够导致模型误判的对抗样本的比例。
                logger.info(f"BCA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")  # 输出攻击的有效性
                
        return adv_x  # 返回对抗样本
    
    def perturb_dae(self, model, purifier, x, label=None,
                steps=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                use_sample=False,
                base=10.,
                verbose=False,
                oblivion=False):
        """
        增强攻击
        """
        # 检查lambda_值的范围是否有效
        assert 0 < min_lambda_ <= max_lambda_
        
        # 将模型设为评估模式，这意味着模型中的所有dropout和batchnorm层将工作在评估模式  
        model.eval() 
        
        # 如果模型有属性is_detector_enabled
        if hasattr(model, 'is_detector_enabled'):  
            self.lambda_ = min_lambda_  # 则将lambda_设置为min_lambda_
        else:
            self.lambda_ = max_lambda_  # 否则将lambda_设置为max_lambda_
            
        # 创建输入x的副本，用于存储对抗样本的数据
        adv_x = x.detach().clone().to(torch.double) 
        
        # 当lambda_值小于或等于max_lambda_时，继续进行循环
        while self.lambda_ <= max_lambda_:  
            with torch.no_grad():  # 不需要计算梯度
                if not oblivion:
                    purified_adv = purifier(adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_loss(model, purified_adv, label, self.lambda_)
                else:
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)
                
            if torch.all(done):  # 如果所有的样本都已完成，即完成标志都为True
                break  # 则跳出循环
            
            pert_x = self._perturb(model, adv_x[~done], label[~done],  # 否则，对未完成的样本进行扰动
                                   steps,
                                   lmda=self.lambda_,
                                   use_sample=use_sample
                                   )
            
            adv_x[~done] = pert_x  # 更新对抗样本的数据
            self.lambda_ *= base  # 调整lambda_的值
            
        with torch.no_grad():  # 不需要计算梯度
            purified_adv = purifier(adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lambda_)
            
            if verbose:  # 如果需要输出详细信息
                # 它会打印出攻击的有效性，即成功生成能够导致模型误判的对抗样本的比例。
                logger.info(f"BCA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")  # 输出攻击的有效性
                
        return adv_x  # 返回对抗样本