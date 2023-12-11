import torch

import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.Pointwise')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120

from tqdm import tqdm

class Pointwise(BaseAttack):
    """
    Pointwise attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param ben_x: torch.FloatTensor, feature vectors with shape [number_of_benign_files, vocab_dim]
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_x, oblivion=False, device=None):
        super(Pointwise, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, repetition=10, max_eta=1, epsilon=1000, seed=0, is_apk=False, verbose=False):
        # 确保重复次数大于0
        assert repetition > 0
        
        # 若x为空或长度小于等于0，则返回空列表
        if x is None or len(x) <= 0:
            return []
        
        # 如果self.ben_x为空，则直接返回x
        if len(self.ben_x) <= 0:
            return x
        
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flags = np.array([])  # 初始化成功标志数组
        x_mod_list = []          # 初始化修改的特征向量列表
        x_adv_init = []
        
        
        # For the pointwise attack, we first continuously enhance the generation of salt-and-pepper noises 
        # and add them to the malicious sample, increasing the noise intensity by 1/1000 each time 
        # until the classifier misclassifies it as benign or Repeat the process 10 times to generate 
        # adversarial samples by modifying the features. 
        # Pointwise attack strategy
        def salt_and_pepper_noise(x, model, max_eta=1, repetition=10, epsilon=1000):
            device = model.device
            torch.manual_seed(seed)

            perturbed_x = x.clone()
            for _ in range(repetition):
                use_flag = False
                for eta in torch.linspace(0, max_eta, epsilon):
                    # print("eta:", eta)
                    uni_noises = torch.rand(perturbed_x.shape).to(device)
                    salt = (uni_noises >= 1. - eta / 2).float()     # salt 噪声
                    pepper = -(uni_noises < eta / 2).float()        # pepper 噪声
                    
                    perturbed_x += salt + pepper
                    perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)
                    
                # 转换为模型需要的输入格式
                perturbed_x, y = utils.to_tensor(perturbed_x.double(), torch.ones(trials,).long(), device)
                
                # 使用模型进行推断
                y_cent, x_density = model.inference_batch_wise(perturbed_x)
                y_pred = np.argmax(y_cent, axis=-1)    

                # 判断攻击是否成功
                if hasattr(model, 'indicator') and not self.oblivion:
                    use_flag = (y_pred == 0) & model.indicator(x_density, y_pred)
                else:
                    use_flag = (y_pred == 0)
                    
                if self.oblivion:
                    continue
                elif use_flag:
                    break
                
            return perturbed_x
        

        with torch.no_grad():
            x_adv_init = [salt_and_pepper_noise(xi, model, max_eta, repetition, epsilon) for xi in x]
            x_adv_init = torch.stack(x_adv_init).detach().cpu().numpy()
        
        # 用于存储对抗样本的数据
        adv_x_list = []
        with torch.no_grad():  # 不进行梯度计算，这在评估和预测时很有用，可以节省内存和加速
            torch.manual_seed(seed)  
            x_adv_init = torch.tensor(np.array(x_adv_init))     # 首先将列表转为numpy数组，然后转换为张量
            x_adv = x_adv_init.clone()                          # 创建对抗样本的初始版本

            # 用来存储每个样本是否成功生成对抗样本的标记
            success_flag = []
            
            x_len = len(x)
            # 遍历原始样本和其对应的对抗样本初始化
            for idx, (original, adversarial) in enumerate(tqdm(zip(x, x_adv_init), total=x_len, desc="Attacking")):
                original = original.flatten()       # 将原始数据平铺为一维
                adversarial = adversarial.flatten()  # 将对抗数据平铺为一维

                found = False  # 设置标记为False，表示还未找到有效的对抗样本
                indices = torch.randperm(len(original))  # 随机打乱特征的索引

                for i in indices:
                    tmp_value = adversarial[i]      # 保存当前位置的对抗样本值
                    adversarial[i] = original[i]    # 尝试将对抗样本在该索引处的值替换为原始样本在相同位置的值
                    
                    # 将改变后的对抗样本转为张量并添加一个新的维度
                    adversarial_tensor = adversarial.clone().detach().to(model.device).to(torch.float64)
        
                    # 使用predict_model模型对清洗后的对抗样本进行预测
                    y_cent, x_density = model.inference_batch_wise(adversarial_tensor)
                    prediction = np.argmax(y_cent, axis=-1)
                    
                    if hasattr(model, 'indicator') and not self.oblivion:
                        use_flag = (prediction == 0) & model.indicator(x_density, prediction)
                    else:
                        use_flag = (prediction == 0)

                    # 检查模型预测结果
                    if use_flag:  # 假设0表示不正确的分类
                        found = True  # 找到有效的对抗样本
                        break
                    else:
                        adversarial[i] = tmp_value  # 如果不成功，撤销此更改

                success_flag.append(found)
                x_adv[idx] = adversarial.reshape(x_adv[idx].shape)

            x_adv_tensor = torch.stack([item.clone().detach().to(x.device) for item in x_adv])
            x_mod = (x_adv_tensor - x).cpu().numpy()
            x_mod_list.append(x_mod)
            adv_x_list.append(x_adv_tensor.detach().cpu().numpy())

        # 根据is_apk返回相应的结果
        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None

        

    def perturb_dae(self, dae_model, predict_model, x, trials=10, repetition=10, max_eta=1, epsilon=1000, seed=0, is_apk=False, verbose=False):
        # 确保重复次数大于0
        assert trials > 0
        
        # 若x为空或长度小于等于0，则返回空列表
        if x is None or len(x) <= 0:
            return []
        
        # 如果self.ben_x为空，则直接返回x
        if len(self.ben_x) <= 0:
            return x
        
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flags = np.array([])  # 初始化成功标志数组
        x_mod_list = []          # 初始化修改的特征向量列表
        x_adv_init = []
        
        # For the pointwise attack, we first continuously enhance the generation of salt-and-pepper noises 
        # and add them to the malicious sample, increasing the noise intensity by 1/1000 each time 
        # until the classifier misclassifies it as benign or Repeat the process 10 times to generate 
        # adversarial samples by modifying the features. 
        # Pointwise attack strategy
        def salt_and_pepper_noise(x, model, max_eta=1, repetition=10, epsilon=1000):
            device = model.device
            torch.manual_seed(seed)

            perturbed_x = x.clone()
            for _ in range(repetition):
                use_flag = False
                for eta in torch.linspace(0, max_eta, epsilon):
                    # print("eta:", eta)
                    uni_noises = torch.rand(perturbed_x.shape).to(device)
                    salt = (uni_noises >= 1. - eta / 2).float()     # salt 噪声
                    pepper = -(uni_noises < eta / 2).float()        # pepper 噪声
                    
                    perturbed_x += salt + pepper
                    perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)
                    
                # 转换为模型需要的输入格式
                perturbed_x, y = utils.to_tensor(perturbed_x.double(), torch.ones(trials,).long(), device)
                
                # 使用模型进行推断
                y_cent, x_density = predict_model.inference_batch_wise(perturbed_x)
                y_pred = np.argmax(y_cent, axis=-1)    

                # 判断攻击是否成功
                if hasattr(model, 'indicator') and not self.oblivion:
                    use_flag = (y_pred == 0) & model.indicator(x_density, y_pred)
                else:
                    use_flag = (y_pred == 0)
                    
                if self.oblivion:
                    continue
                elif use_flag:
                    break
                
            return perturbed_x
        
        success_flags = np.array(success_flags)  # 转换为numpy数组

        with torch.no_grad():
            x_adv_init = [salt_and_pepper_noise(xi, dae_model, max_eta, repetition, epsilon) for xi in x]
            x_adv_init = torch.stack(x_adv_init).detach().cpu().numpy()

        # 用于存储对抗样本的数据
        adv_x_list = []        
        with torch.no_grad():  # 不进行梯度计算，这在评估和预测时很有用，可以节省内存和加速
            torch.manual_seed(seed)  # 为了确保实验的可重复性
            x_adv_init = torch.tensor(np.array(x_adv_init))  # 首先将列表转为numpy数组，然后转换为张量
            x_adv = x_adv_init.clone()  # 创建对抗样本的初始版本

            # 用来存储每个样本是否成功生成对抗样本的标记
            success_flag = []

            x_len = len(x)
            # 遍历原始样本和其对应的对抗样本初始化
            for idx, (original, adversarial) in enumerate(zip(x, x_adv_init)):
                original = original.flatten()       # 将原始数据平铺为一维
                adversarial = adversarial.flatten()  # 将对抗数据平铺为一维

                found = False  # 设置标记为False，表示还未找到有效的对抗样本
                indices = torch.randperm(len(original))  # 随机打乱特征的索引

                for i in indices:
                    tmp_value = adversarial[i]      # 保存当前位置的对抗样本值
                    adversarial[i] = original[i]    # 尝试将对抗样本在该索引处的值替换为原始样本在相同位置的值
                    
                    # 将改变后的对抗样本转为张量并添加一个新的维度
                    adversarial_tensor = adversarial.clone().detach().to(dae_model.device).to(torch.float32).unsqueeze(0)
        
                    # 使用denoising autoencoder模型清洗对抗样本
                    Purified_adv_x_batch = dae_model(adversarial_tensor)
                    
                    # 转换模型输出为numpy数组并根据阈值进行二值化
                    outputs_numpy = Purified_adv_x_batch.cpu().numpy()
                    reshape_encoded_data = np.where(outputs_numpy >= 0.5, 1, 0)
                    
                    # 将二值化后的数据转回张量
                    Purified_modified_x = torch.tensor(reshape_encoded_data, device=dae_model.device, dtype=torch.float64).squeeze(0)
                    
                    # 使用predict_model模型对清洗后的对抗样本进行预测
                    y_cent, x_density = predict_model.inference_batch_wise(Purified_modified_x)
                    prediction = np.argmax(y_cent, axis=-1)

                    # 检查模型预测结果
                    if prediction == 0:  # 假设0表示不正确的分类
                        found = True  # 找到有效的对抗样本
                        break
                    else:
                        adversarial[i] = tmp_value  # 如果不成功，撤销此更改

                success_flag.append(found)
                x_adv[idx] = adversarial.reshape(x_adv[idx].shape)

            x_adv_tensor = torch.stack([item.clone().detach().to(x.device) for item in x_adv])
            x_mod = (x_adv_tensor - x).cpu().numpy()
            x_mod_list.append(x_mod)
            adv_x_list.append(x_adv_tensor.detach().cpu().numpy())

        # 根据is_apk返回相应的结果
        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None
