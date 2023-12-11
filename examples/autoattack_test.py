from __future__ import absolute_import  # 使用绝对引入，即使存在同名模块也不会引入
from __future__ import division  # 除法运算使用Python3的行为，即总是返回浮点数
from __future__ import print_function  # 强制使用print函数，而不是Python2的print语句
import os
import argparse  # 引入命令行参数解析库

import numpy as np
import torch
import torch.nn as nn

from core.defense import Dataset  # 导入防御模块中的Dataset类

from core.defense import MalwareDetectionDNN, PGDAdvTraining, RFGSMAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE, VAE_SU
    
from torch.utils.data import DataLoader, TensorDataset
    
from autoattack import AutoAttack

from tools import utils  # 导入工具模块
from config import config, logging, ErrorHandler  # 导入配置模块

logger = logging.getLogger('examples.autoattack_test')  # 获取日志记录器
logger.addHandler(ErrorHandler)  # 添加错误处理器

atta_argparse = argparse.ArgumentParser(description='arguments for autoattack')  # 创建命令行参数解析器

atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='每个批次加载的示例数。')  # 添加命令行参数

atta_argparse.add_argument('--norm', type=str, default='L2',
                           choices=['Linf', 'L2', 'L1'],
                           help="norm = ['Linf' | 'L2' | 'L1'] is the norm of the threat model")

atta_argparse.add_argument('--eps', type=float, default=1.,
                           help="eps is the bound on the norm of the adversarial perturbations")

atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus',
                                    'amd_pad_ma', 'fd_vae', 'dae'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae', 'dae'.")

atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')
atta_argparse.add_argument('--basic_dnn_name', type=str, default='20230724-230516',
                           help='basic_dnn_name')


# 定义一个适配函数，将10000*1的向量调整为1*1*10000的张量
def adapt_input(x):
    x = x.reshape(-1, 1, 1, 10000)
    return x

class AdaptedModel(nn.Module):
    def __init__(self, original_model):
        super(AdaptedModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        # 首先使用适配函数将输入调整为适合AutoAttack的形状
        x = adapt_input(x)
        
        # 将数据flatten
        x = x.view(x.size(0), -1)
        
        # print("调用原始模型进行推断: x.shape", x.shape)
        x = x.to(torch.float32)
        
        # 然后调用原始模型进行推断
        logits = self.original_model.forward(x)
        # print("logits.shape", logits.shape)
        if isinstance(logits, tuple):
            return logits[0]
        else:
            return logits

def _main():
    args = atta_argparse.parse_args()  # 解析命令行参数
    
    # 根据输入的模型参数选择对应的模型，并设定保存目录
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'md_at_pgd':
        save_dir = config.get('experiments', 'md_at_pgd') + '_' + args.model_name
    elif args.model == 'md_at_ma':
        save_dir = config.get('experiments', 'md_at_ma') + '_' + args.model_name
    elif args.model == 'md_at_fgsm':
        save_dir = config.get('experiments', 'md_at_fgsm') + '_' + args.model_name
    elif args.model == 'amd_kde':
        save_dir = config.get('experiments', 'amd_kde') + '_' + args.model_name
    elif args.model == 'amd_icnn':
        save_dir = config.get('experiments', 'amd_icnn') + '_' + args.model_name
    elif args.model == 'amd_dla':
        save_dir = config.get('experiments', 'amd_dla') + '_' + args.model_name
    elif args.model == 'amd_dnn_plus':
        save_dir = config.get('experiments', 'amd_dnn_plus') + '_' + args.model_name
    elif args.model == 'amd_pad_ma':
        save_dir = config.get('experiments', 'amd_pad_ma') + '_' + args.model_name
    elif args.model == 'fd_vae':
        save_dir = config.get('experiments', 'fd_vae') + '_' + args.model_name
    elif args.model == 'dae':
        save_dir = config.get('experiments', 'dae') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae' and 'dae'.")

    # 读取超参数配置
    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))

    # 创建数据集
    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
    test_x, testy = dataset.test_dataset

    # 获取恶意样本并保存，如果已存在则直接读取
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)

    # 打印出总共恶意样本的数量
    logger.info(f"Total number of malicious samples: {len(mal_test_x)}")
    
    # 检查恶意样本数量
    mal_count = len(mal_testy)
    if mal_count <= 0:
        return

    # 创建输入生产器
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=args.batch_size,
                                                           name='test')

    # 检查类别数
    assert dataset.n_classes == 2

    # 检测是否使用CUDA，不使用则使用CPU
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
     
    from pprint import pprint   
    pprint(hp_params)
    pprint(args)
        
    # 创建并加载模型
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    # 根据模型类型创建对应的模型
    # 如果模型是ICNN或者PAD-MA则添加进阶处理
    # 根据不同模型类型，调整模型参数并加载模型
    # 其他类型直接加载模型
    
    if args.model == 'amd_icnn' or args.model == 'amd_pad_ma':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       name=args.model_name,
                                       **hp_params
                                       )
    # 将模型移动到指定设备（GPU或者CPU）并转为double类型
    model = model.to(dv)

    # 对不同的模型类型，执行不同的加载策略
    if args.model == 'md_at_pgd':
        at_wrapper = PGDAdvTraining(model)
        at_wrapper.load()  # 加载PGD对抗训练模型
        model = at_wrapper.model
        
    elif args.model == 'md_at_ma':
        at_wrapper = MaxAdvTraining(model)
        at_wrapper.load()  # 加载Max对抗训练模型
        model = at_wrapper.model
        
    elif args.model == 'md_at_fgsm':
        at_wrapper = RFGSMAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model  
                
    elif args.model == 'amd_kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()  # 加载KernelDensityEstimation模型
        
    elif args.model == 'amd_dla':
        model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=args.model_name,
                                     **hp_params
                                     )
        model = model.to(dv)
        model.load()  # 加载AMalwareDetectionDLA模型
        
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv)
        model.load()  # 加载AMalwareDetectionDNNPlus模型
        
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()  # 加载AMalwareDetectionPAD模型
        model = adv_model.model

    elif args.model == 'fd_vae':
        model = VAE_SU(name=args.model_name)
        model.load()
        model.to(dv)

    elif args.model == 'dae':
        model = DAE(input_size = dataset.vocab_size,
            device = dv,
            name = args.model_name,
            predict_model_name = args.basic_dnn_name
            )
        model.load()
        model.to(dv)
        
        predict_model = MalwareDetectionDNN(dataset.vocab_size,
                            dataset.n_classes,
                            device=dv,
                            name=model.predict_model_name,
                            **hp_params
                            )
        predict_model = predict_model.to(dv)
        predict_model_save_path = os.path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        print("Basic DNN Model: ", predict_model_save_path)
        predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()
        
    else:
        model.load()  # 对其他类型的模型直接加载模型

    # 使用保存的模型参数
    logger.info("Load model parameters from {}.".format(model.model_save_path))

    # 初始化几个空列表用于后续收集数据
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    
    x_adv_samples = []

    # 设置模型为评估模式
    model.eval()
    
    adapted_model = AdaptedModel(model)
    
    # 初始化 AutoAttack
    adversary = AutoAttack(adapted_model, norm=args.norm, eps=args.eps, version='standard') # 创建AutoAttack对抗者

    # 攻击方式 (3种组合)
    adversary.attacks_to_run = ['apgd-ce', 'fab'] # 'apgd-ce', 'fab', 'square'
    adversary.apgd.n_restarts = 2
    adversary.fab.n_restarts = 2

    if args.model == 'dae':
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.float(), y.long(), model.device)
            
            adapted_model = AdaptedModel(predict_model)
            
            # 初始化 AutoAttack
            adversary = AutoAttack(adapted_model, norm=args.norm, eps=args.eps, version='standard') # 创建AutoAttack对抗者

            # 攻击方式 (3种组合)
            adversary.attacks_to_run = ['apgd-ce', 'fab'] # 'apgd-ce', 'fab', 'square'
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2
            adv_x_batch = adversary.run_standard_evaluation(x, y, args.batch_size) 

            # 对抗样本的数据类型转换
            adv_x_batch = adv_x_batch.to(torch.float32)

            # 使用当前模型清洗对抗样本
            Purified_adv_x_batch = model(adv_x_batch).to(torch.float32)

            Purified_adv_x_batch = Purified_adv_x_batch.to(model.device)
            
            # 使用预测模型对清洗后的对抗样本进行预测
            y_cent_batch, _ = predict_model.inference_batch_wise(Purified_adv_x_batch)
            y_cent_list.append(y_cent_batch)

        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
        logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / len(y_pred) * 100:.3f}%')
    
    else:
        # 对测试集中的每一个样本进行处理
        for x, y in mal_test_dataset_producer:
            # 将数据转为张量
            x, y = utils.to_tensor(x.float(), y.long(), model.device)

            # 对模型进行对抗攻击并得到对抗样本
            adv_x_batch = adversary.run_standard_evaluation(x, y, args.batch_size) 

            # 对扰动后的数据进行推理
            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)

            # 收集数据
            y_cent_list.append(y_cent_batch)
            x_density_list.append(x_density_batch)
            x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
            x_adv_samples.append((adv_x_batch).detach().cpu().numpy())

        # 求出预测结果
        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)

        # 输出对抗样本的平均准确率
        # 这里的准确率是指模型成功地将这些样本预测为恶意软件的比例。
        logger.info(
            f'The mean accuracy on autoattack is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    # 如果模型有指示器，计算指示器的效果
    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        # 计算并输出指示器的有效性
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        # 计算并输出模型对对抗性恶意软件样本的预测准确率，这里考虑了指示器的结果。
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

if __name__ == '__main__':
    _main()