"""
base class for waging attacks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import multiprocessing

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

from core.droidfeature import InverseDroidFeature
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('examples.base_attack')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BaseAttack(Module):
    """
    攻击的抽象类

    参数
    ---------
    @param is_attacker, Boolean, 扮演攻击者的角色（注意：防御者进行对抗性训练）
    @param oblivion, Boolean, 是否知道敌手的指标
    @param kappa, float, 攻击信心值
    @param manipulation_x, boolean 向量显示可修改的APIs
    @param omega, 由4个集合组成的列表，每个集合都包含与每个api对应的相互依赖的api的索引
    @param device, 'cpu' 或 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        # 调用父类的构造函数
        super(BaseAttack, self).__init__()
        
        # 是否是攻击者
        self.is_attacker = is_attacker
        
        # 是否知道对手的指标
        self.oblivion = oblivion
        
        # 攻击的信心值
        self.kappa = kappa
        
        # 可修改的APIs
        self.manipulation_x = manipulation_x
        
        # 运行设备，CPU或CUDA（GPU）
        self.device = device
        
        # 与每个api对应的相互依赖的api的索引集合
        self.omega = omega
        
        # 反向特征的对象
        self.inverse_feature = InverseDroidFeature()
        
        # 进行初始化操作
        self.initialize()

    def initialize(self):
        # 判断是否指定了可操作的APIs
        if self.manipulation_x is None:
            # 未指定时，从inverse_feature获取默认的可操作APIs
            self.manipulation_x = self.inverse_feature.get_manipulation()

        # 将manipulation_x转为LongTensor类型，并移动到指定的设备上（CPU或GPU）
        self.manipulation_x = torch.LongTensor(self.manipulation_x).to(self.device)

        # 判断是否指定了与每个api对应的相互依赖的api的索引
        if self.omega is None:
            # 未指定时，从inverse_feature获取默认的相互依赖的APIs
            self.omega = self.inverse_feature.get_interdependent_apis()

        # 使用one_hot编码处理self.omega，并计算每列的和，将结果存入self.omega中
        # 这样每个API的值就表示它依赖于哪些APIs
        self.omega = torch.sum(
            F.one_hot(torch.tensor(self.omega), num_classes=len(self.inverse_feature.vocab)),
            dim=0).to(self.device)

        # 获取API标志位
        api_flag = self.inverse_feature.get_api_flag()
        
        # 将API标志位转为布尔类型的LongTensor，并移动到指定的设备上
        self.api_flag = torch.LongTensor(api_flag).bool().to(self.device)


    def perturb(self, model, x, adj=None, label=None):
        """
        扰动节点特征向量

        参数
        --------
        @param model: 被攻击的模型
        @param x: torch.FloatTensor, 节点特征向量，每个都代表一个API
        @param adj: torch.FloatTensor或None, 邻接矩阵（如果不为None，则其形状是[number_of_graphs, batch_size, vocab_dim, vocab_dim]）
        @param label: torch.LongTensor, 真实的标签
        """
        
        # 这是一个抽象方法，需要在子类中进行具体实现
        raise NotImplementedError

    
    def produce_adv_mal(self, x_mod_list, feature_path_list, app_dir, save_dir=None):
        """
        在实践中生成对抗性恶意软件。

        参数
        --------
        @param x_mod_list: tensor的列表，每个tensor对应于应用于特性的数值修改。
        @param feature_path_list: 特征路径的列表，每个路径对应于调用图保存的文件。
        @param app_dir: 字符串，指向原始恶意应用的目录（或路径列表）。
        @param save_dir: 用于保存生成的APK的目录。
        """
        
        if len(x_mod_list) <= 0:
            return

        assert len(x_mod_list) == len(feature_path_list)
        assert isinstance(x_mod_list[0], (torch.Tensor, np.ndarray))

        # 如果未指定保存目录，则默认为'/tmp/adv_mal_cache'
        if save_dir is None:
            save_dir = os.path.join('/tmp/', 'adv_mal_cache')

        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)

        # 将修改转换为具体的指令
        x_mod_instructions = [self.inverse_feature.inverse_map_manipulation(x_mod) 
                            for x_mod in x_mod_list]

        # 根据提供的应用目录或应用路径列表，获取具体的应用路径
        if os.path.isdir(app_dir):
            app_path_list = [os.path.join(app_dir, os.path.basename(os.path.splitext(feat_p)[0])) 
                            for feat_p in feature_path_list]
        elif isinstance(app_dir, list):
            app_path_list = app_dir
        else:
            raise ValueError("期望应用目录或路径列表，但得到 {}.".format(type(app_dir)))

        assert np.all([os.path.exists(app_path) for app_path in app_path_list]), "找不到所有的应用路径。"

        # 准备多进程参数
        pargs = [(x_mod_instr, feature_path, app_path, save_dir) 
                for x_mod_instr, feature_path, app_path in zip(x_mod_instructions, 
                                                            feature_path_list, app_path_list) 
                if not os.path.exists(os.path.join(save_dir, 
                                                    os.path.splitext(os.path.basename(app_path))[0] + '_adv'))]

        # 设置进程数，至少为1
        cpu_count = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() - 2 > 1 else 1
        pool = multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer)

        # 并行处理，按顺序保持
        for res in pool.map(InverseDroidFeature.modify_wrapper, pargs):  
            if isinstance(res, Exception):
                logger.exception(res)

        pool.close()
        pool.join()


    def check_lambda(self, model):
        """
        检查模型是否具有检测功能并是否知道关于对手的信息。
        """
        
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            return True
        else:
            return False



    def get_loss(self, model, adv_x, label, lambda_=None):
        # 如果模型有'is_detector_enabled'属性，说明模型不仅仅是分类器，还有检测器的功能。
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(adv_x)
        else:
            # 否则，我们只从模型获取分类的logits
            logits_f = model.forward(adv_x)
        
        # print(type(logits_f))
        # print("logits_f.shape:", logits_f.shape)

        # 计算交叉熵损失，其中'reduction='none''意味着对每个样本都计算损失，而不是求平均
        ce = F.cross_entropy(logits_f, label, reduction='none')
        
        # 获取模型的预测类别
        y_pred = logits_f.argmax(1)
        
        # 如果模型有检测器功能，并且我们没有选择'oblivion'(即我们知道对方是否是对手)
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            assert lambda_ is not None
            # 获取每个样本的tau值
            tau = model.get_tau_sample_wise(y_pred)
            
            # 根据是否为攻击者，计算不同的损失
            if self.is_attacker:
                loss_no_reduction = ce + lambda_ * (torch.clamp(tau - prob_g, max=self.kappa))
            else:
                loss_no_reduction = ce + lambda_ * (tau - prob_g)
            
            # 判断哪些样本是被成功攻击的
            done = (y_pred != label) & (prob_g <= tau)
        else:
            # 如果没有检测器功能，损失只是交叉熵
            loss_no_reduction = ce
            # 判断哪些样本是被成功攻击的
            done = y_pred != label
        
        return loss_no_reduction, done
    

    def get_scores(self, model, pertb_x, label, lmda=1.):
        # 如果模型有'is_detector_enabled'属性，说明模型不仅仅是分类器，还有检测器的功能。
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(pertb_x)
        else:
            # 否则，我们只从模型获取分类的logits
            logits_f = model.forward(pertb_x)
        
        # 获取模型的预测类别
        y_pred = logits_f.argmax(1)
        
        # 如果模型有检测器功能，并且我们没有选择'oblivion'
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            # 获取每个样本的tau值
            tau = model.get_tau_sample_wise(y_pred)
            # 判断哪些样本是被成功攻击的
            done = (y_pred != label) & (prob_g <= tau)
        else:
            # 判断哪些样本是被成功攻击的
            done = y_pred != label
        
        return done