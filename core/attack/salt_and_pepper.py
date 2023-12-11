import torch

import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.salt_and_pepper')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120



class Salt_and_pepper(BaseAttack):
    def __init__(self, ben_x, oblivion=False, device=None):
        super(Salt_and_pepper, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

        
    def perturb(self, model, x, trials=10, epsilon=10, max_eta=0.001, repetition=10, seed=0, is_apk=False, verbose=False):
        # 如果输入x为空或长度小于等于0，返回空列表
        if x is None or len(x) <= 0:
            return []

        # 如果self.ben_x长度小于等于0，直接返回x
        if len(self.ben_x) <= 0:
            return x

        # trials参数不能超过self.ben_x的长度
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)

        # 获取模型所在的设备（例如：CPU或CUDA）
        device = model.device       # 假设模型有一个device属性
        torch.manual_seed(seed)     # 设置随机种子

        success_flags = []          # 用于记录每次尝试是否成功的标志
        x_mod_list = []             # 用于记录每次尝试修改后的x
        
        # 用于存储对抗样本的数据
        adv_x_list = []

        with torch.no_grad():  # 不计算梯度，提高效率
            for _x in x:
                shape = _x.shape
                perturbed_x = _x.clone()  # 复制原始x

                # 对x重复添加噪声
                for _ in range(repetition):
                    for eta in torch.linspace(0, max_eta, steps=min(epsilon, shape[0]))[1:]:
                        # 生成随机噪声
                        uni_noises = torch.rand(perturbed_x.shape).to(device)
                        salt = (uni_noises >= 1. - eta / 2).float()     # salt 噪声
                        pepper = -(uni_noises < eta / 2).float()        # pepper 噪声

                        # 将噪声添加到x上
                        perturbed_x += salt + pepper
                        
                        # 保证x的值在0和1之间
                        perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)
                        perturbed_x = utils.round_x(perturbed_x, 0.5)  # 对x进行四舍五入处理                        

                        if hasattr(model, 'indicator') and not self.oblivion:
                            y_cent, x_density = model.inference_batch_wise(perturbed_x)
                            use_flag = (y_pred == 0) & model.indicator(x_density, y_pred)
                            if use_flag:
                                break

                    # 转换为模型需要的输入格式
                    #perturbed_x, y = utils.to_tensor(perturbed_x.double(), torch.ones(trials,).long(), device)
                    
                    # 使用模型进行推断
                    y_cent, x_density = model.inference_batch_wise(perturbed_x)
                    y_pred = np.argmax(y_cent, axis=-1)

                    # 判断攻击是否成功
                    if hasattr(model, 'indicator') and not self.oblivion:
                        use_flag = (y_pred == 0) & model.indicator(x_density, y_pred)
                    else:
                        use_flag = (y_pred == 0)
                        
                    if use_flag:
                        break
                  
                # print("判断攻击是否成功 - use_flag:", use_flag)

                # 记录攻击是否成功的标志
                success_flags.append(use_flag)

                # 记录修改后的x
                x_mod = (perturbed_x - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)
                adv_x_list.append(perturbed_x.detach().cpu().numpy())

        success_flags = np.array(success_flags)  # 转换为numpy数组

        # 根据is_apk返回相应的结果
        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None
        
    