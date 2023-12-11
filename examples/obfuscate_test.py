from __future__ import absolute_import  # 使用绝对引入，即使存在同名模块也不会引入
from __future__ import division  # 除法运算使用Python3的行为，即总是返回浮点数
from __future__ import print_function  # 强制使用print函数，而不是Python2的print语句
import os
import argparse  # 引入命令行参数解析库

import numpy as np
import torch
import joblib
import pickle

from core.defense import Dataset  # 导入防御模块中的Dataset类

from core.defense import MalwareDetectionDNN, PGDAdvTraining, RFGSMAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE, VAE_SU, \
    MalwareDetectionCNN, MalwareDetectionDT, MalwareDetectionFCN, MalwareDetectionLSTM, MalwareDetectionRF, MalwareDetectionRNN, MalwareDetectionSVM
    
from torch.utils.data import DataLoader, TensorDataset
    
from core.attack import BCA  # 导入攻击模块中的BCA类
from tools import utils  # 导入工具模块
from config import config, logging, ErrorHandler  # 导入配置模块

logger = logging.getLogger('examples.obfuscate_test')  # 获取日志记录器
logger.addHandler(ErrorHandler)  # 添加错误处理器

# example:    python -m examples.bca_test --steps $i --model "amd_pad_ma" --model_name "20230717-195459"

atta_argparse = argparse.ArgumentParser(description='arguments for bca')  # 创建命令行参数解析器

atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='是否知道对手指示器。')  # 添加命令行参数

atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='攻击置信度。')  # 添加命令行参数

atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='每个批次加载的示例数。')  # 添加命令行参数

atta_argparse.add_argument('--Ob_type', type=str, default='COMBINE',
                           choices=['API_REFLECTION', 'BENIGN_CLASS', 'COMBINE', 'PCM',
                                    'RESOURCE', 'RM_PERMISSION', 'STRING', 'VARIABLE'],
                           help="混淆方式: 'API_REFLECTION', 'BENIGN_CLASS', 'COMBINE', 'PCM',\
                                    'RESOURCE', 'RM_PERMISSION', 'STRING', 'VARIABLE' ")

atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus',
                                    'amd_pad_ma', 'fd_vae', 'dae',
                                    'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae', 'dae', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'.")

atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')

atta_argparse.add_argument('--basic_model', type=str, default='md_dnn',
                           choices=['md_dnn', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'],
                           help="'md_dnn', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'")

atta_argparse.add_argument('--basic_model_name', type=str, default='20230724-230516',
                           help='dnn basic_dnn_name')


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
    elif args.model == 'rf':
        save_dir = config.get('experiments', 'rf') + '_' + args.model_name
    elif args.model == 'dt':
        save_dir = config.get('experiments', 'dt') + '_' + args.model_name
    elif args.model == 'md_cnn':
        save_dir = config.get('experiments', 'md_cnn') + '_' + args.model_name         
    elif args.model == 'md_svm':
        save_dir = config.get('experiments', 'md_svm') + '_' + args.model_name    
    elif args.model == 'md_rnn':
        save_dir = config.get('experiments', 'md_rnn') + '_' + args.model_name   
    elif args.model == 'md_lstm':
        save_dir = config.get('experiments', 'md_lstm') + '_' + args.model_name  
    elif args.model == 'md_fcn':
        save_dir = config.get('experiments', 'md_fcn') + '_' + args.model_name                                         
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

    # load pkl files
    obfuscated_apk_file = config.get('DEFAULT', 'project_root') + "/obfuscated/" + config.get('DEFAULT', 'dataset_name') + "/x_" + args.Ob_type + ".pkl"
    
    with open(obfuscated_apk_file, "rb") as fr:
        mal_test_x = joblib.load(fr)

        print("攻击样本总数", len(mal_test_x))
        mal_test_x = np.array(mal_test_x)

    test_z_shape = mal_test_x.shape[0]    # Get the number of samples in test_z

    # Create label sets with all elements set to 1
    mal_test_y = np.ones(test_z_shape, dtype=int)
    
    print("mal_test_x.shape", mal_test_x.shape)
    print("mal_testy.shape", mal_test_y.shape)

    def create_dataloader(mal_test_x, mal_test_y, batch_size):
        # 将numpy数组转为PyTorch张量
        mal_test_x_tensor = torch.tensor(mal_test_x, dtype=torch.float32)
        mal_testy_tensor = torch.tensor(mal_test_y, dtype=torch.int64)
        
        # # 输出形状，以便于调试
        # print(mal_test_x_tensor.shape)
        # print(mal_testy_tensor.shape)
        
        # 使用TensorDataset组合特征和标签
        dataset = TensorDataset(mal_test_x_tensor, mal_testy_tensor)
        
        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader

    mal_test_dataset_producer = create_dataloader(mal_test_x, mal_test_y, args.batch_size)

    # 打印出总共恶意样本的数量
    logger.info(f"Total number of malicious samples: {len(mal_test_x)}")
    
    # 检查恶意样本数量
    mal_count = len(mal_test_y)
    if mal_count <= 0:
        return

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
    model = model.to(dv).double()

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
        model = model.to(dv).double()
        model.load()  # 加载AMalwareDetectionDLA模型
        
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv).double()
        model.load()  # 加载AMalwareDetectionDNNPlus模型
        
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()  # 加载AMalwareDetectionPAD模型
        model = adv_model.model

    elif args.model == 'fd_vae':
        model = VAE_SU(name=args.model_name)
        model.load()
        model.to(dv)
        
    elif args.model == 'rf':
        model = MalwareDetectionRF(name=args.model_name)
        model.load()

    elif args.model == 'dt':
        model = MalwareDetectionDT(name=args.model_name)
        model.load()
        
    elif args.model == 'md_lstm':
        model = MalwareDetectionLSTM(input_dim = dataset.vocab_size,
                                     device = dv,
                                     name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)          
        
    elif args.model == 'md_cnn':
        model = MalwareDetectionCNN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)        
        
    elif args.model == 'md_fcn':
        model = MalwareDetectionFCN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)     
        
    elif args.model == 'md_svm':
        model = MalwareDetectionSVM(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)         
    
    elif args.model == 'md_rnn':
        model = MalwareDetectionRNN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)                       

    elif args.model == 'dae':
        model = DAE(input_size = dataset.vocab_size,
            device = dv,
            name = args.model_name,
            predict_model_name = args.basic_model_name
            )
        model.load()
        model.to(dv)
        
        if args.basic_model == 'md_dnn':
            predict_model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=model.predict_model_name,
                                **hp_params
                                )
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        
        elif args.basic_model == 'rf':
            predict_model = MalwareDetectionRF(name=model.predict_model_name)
            predict_model.load()
            predict_model_save_path = os.path.join(config.get('experiments', 'rf') + '_' + model.predict_model_name,
                                            'model.pkl')
        
        elif args.basic_model == 'dt':
            predict_model = MalwareDetectionDT(name=model.predict_model_name)
            predict_model.load()
            predict_model_save_path = os.path.join(config.get('experiments', 'dt') + '_' + model.predict_model_name,
                                            'model.pkl')

        elif args.basic_model == 'md_svm':
            predict_model = MalwareDetectionSVM(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_svm') + '_' + model.predict_model_name,
                                            'model.pth')          
            
        elif args.basic_model == 'md_lstm':  
            predict_model = MalwareDetectionLSTM(input_dim = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_lstm') + '_' + model.predict_model_name,
                                            'model.pth')    
            
        elif args.basic_model == 'md_cnn':  
            predict_model = MalwareDetectionCNN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_cnn') + '_' + model.predict_model_name,
                                            'model.pth')                
            
        elif args.basic_model == 'md_fcn':
            predict_model = MalwareDetectionFCN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_fcn') + '_' + model.predict_model_name,
                                            'model.pth')      
            
        elif args.basic_model == 'md_rnn':
            predict_model = MalwareDetectionRNN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_rnn') + '_' + model.predict_model_name,
                                            'model.pth')          
            
        print("[⭐] Basic Model: ", predict_model_save_path)
        
        if args.basic_model == 'rf' or args.basic_model == 'dt':
            predict_model.load()
        else:    
            predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()
        
    else:
        model.load()  # 对其他类型的模型直接加载模型

    # 使用保存的模型参数
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    
    # 对测试集进行预测
    if args.model == 'dae':
        model.predict(mal_test_dataset_producer, predict_model, indicator_masking=False)
    else:
        model.predict(mal_test_dataset_producer, indicator_masking=False)

    # 初始化几个空列表用于后续收集数据
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    
    x_adv_samples = []

    # 设置模型为评估模式
    model.eval()

    if args.model == 'dae':
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            
            adv_x_batch = x.detach().clone().to(torch.double) 

            # 对抗样本的数据类型转换
            adv_x_batch = adv_x_batch.to(torch.float32)

            # 使用当前模型清洗对抗样本
            Purified_adv_x_batch = model(adv_x_batch).to(torch.float64)

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
            x, y = utils.to_tensor(x.double(), y.long(), model.device)

            # 对模型进行对抗攻击并得到对抗样本
            adv_x_batch = x.detach().clone().to(torch.double) 

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
            f'The mean accuracy on adversarial malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

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