import torch

import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.mimicry')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120
import random

class Mimicry(BaseAttack):
    """
    Mimicry attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param ben_x: torch.FloatTensor, feature vectors with shape [number_of_benign_files, vocab_dim]
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_x, oblivion=False, device=None):
        super(Mimicry, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, seed=0, is_apk=False, verbose=False):
        """
        修改恶意应用的特征向量。

        参数:
        -----------
        model : PyTorch模型
            受害模型。
        x : torch.FloatTensor
            形状为[batch_size, vocab_dim]的特征向量。
        trials : int
            重复次数。
        seed : int
            随机种子。
        is_apk : bool
            是否生成apk文件。
        verbose : bool
            是否显示攻击信息。

        返回:
        --------
        Tuple
            包含成功标志和修改的特征向量列表或None。
        """

        # 确保重复次数大于0
        assert trials > 0
        
        # 若x为空或长度小于等于0，则返回空列表
        if x is None or len(x) <= 0:
            return []
        
        # 如果self.ben_x为空，则直接返回x
        if len(self.ben_x) <= 0:
            return x
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flag = np.array([])  # 初始化成功标志数组

        with torch.no_grad():
            # torch.manual_seed(torch.manual_seed(int(random.random() * 100)))  # 设置随机种子
            x_mod_list = []          # 初始化修改的特征向量列表

            # 遍历x中的每个元素
            for _x in x:
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                _x_fixed_one = ((1. - self.manipulation_x).float() * _x)[None, :]
                
                # 对x进行修改
                modified_x = torch.clamp(_x_fixed_one + trial_vectors, min=0., max=1.)
                modified_x, y = utils.to_tensor(modified_x.double(), torch.ones(trials,).long(), model.device)
                y_cent, x_density = model.inference_batch_wise(modified_x)
                y_pred = np.argmax(y_cent, axis=-1)
                
                # 根据模型是否有指标属性以及self.oblivion属性判断攻击标志
                if hasattr(model, 'indicator') and (not self.oblivion):
                    attack_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                else:
                    attack_flag = (y_pred == 0)
                
                # 获取最优的ben_id
                ben_id_sel = np.argmax(attack_flag)

                # 检查攻击的有效性
                if 'indicator' in type(model).__dict__.keys():
                    use_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                else:
                    use_flag = attack_flag

                # 根据use_flag更新成功标志
                if not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                else:
                    success_flag = np.append(success_flag, [True])

                # 获取修改后的特征向量与原始特征向量的差
                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)

            # 判断是否需要返回apk
            if is_apk:
                return success_flag, np.vstack(x_mod_list)
            else:
                return success_flag, None
            
    def perturb_dae(self, dae_model, predict_model, x, trials=10, seed=0, is_apk=False, verbose=False):
        # 确保重复次数大于0
        assert trials > 0
        
        # 若x为空或长度小于等于0，则返回空列表
        if x is None or len(x) <= 0:
            return []
        
        # 如果self.ben_x为空，则直接返回x
        if len(self.ben_x) <= 0:
            return x
        
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flag = np.array([])  # 初始化成功标志数组

        with torch.no_grad():
            # torch.manual_seed(torch.manual_seed(int(random.random() * 100)))  # 设置随机种子
            x_mod_list = []          # 初始化修改的特征向量列表

            # 遍历x中的每个元素
            for _x in x:
                
                # print("original x:", _x.cpu().numpy());
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                _x_fixed_one = ((1. - self.manipulation_x).float() * _x)[None, :]
                
                # 对x进行修改
                modified_x = torch.clamp(_x_fixed_one + trial_vectors, min=0., max=1.)
                modified_x, y = utils.to_tensor(modified_x.double(), torch.ones(trials,).long(), predict_model.device)
                
                # 对抗样本的数据类型转换
                adversarial = modified_x.to(torch.float32)
                
                # print("after mimicry modify:", adversarial)
                
                # 使用当前模型清洗对抗样本
                outputs = dae_model(adversarial)              

                # 清洗后的样本数据类型转换
                Purified_modified_x = outputs.to(torch.float64)
                
                modified_x = Purified_modified_x.to(dae_model.device)                
                
                y_cent, x_density = predict_model.inference_batch_wise(modified_x)
                y_pred = np.argmax(y_cent, axis=-1)
                
                # 根据模型是否有指标属性以及self.oblivion属性判断攻击标志
                if hasattr(predict_model, 'indicator') and (not self.oblivion):
                    attack_flag = (y_pred == 0) & (predict_model.indicator(x_density, y_pred))
                else:
                    attack_flag = (y_pred == 0)
                
                # 获取最优的ben_id
                ben_id_sel = np.argmax(attack_flag)

                # 检查攻击的有效性
                if 'indicator' in type(predict_model).__dict__.keys():
                    use_flag = (y_pred == 0) & (predict_model.indicator(x_density, y_pred))
                else:
                    use_flag = attack_flag

                # 根据use_flag更新成功标志
                if not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                else:
                    # print("Mimicary attack success: ", modified_x)
                    success_flag = np.append(success_flag, [True])

                # 获取修改后的特征向量与原始特征向量的差
                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)

            # 判断是否需要返回apk
            if is_apk:
                return success_flag, np.vstack(x_mod_list)
            else:
                return success_flag, None        

