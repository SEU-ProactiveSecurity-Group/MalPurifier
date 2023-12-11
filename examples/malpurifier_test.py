# 使用未来版本特性，确保代码在Python2和Python3中有一致的行为
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入所需的库
import os.path as path
import argparse
import time
import numpy
import torch

# 导入自定义模块
from core.defense import Dataset
from core.defense import malpurifier
from core.defense import MalwareDetectionDNN
from tools.utils import save_args, get_group_args, to_tensor, dump_pickle, read_pickle

# 从config模块中导入配置、日志和错误处理相关功能
from config import config, logging, ErrorHandler

# 初始化argparse对象，用于解析命令行参数
cmd_md = argparse.ArgumentParser(description='arguments for learning malware detector')

# 定义与特征提取相关的命令行参数
feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=2,
                              help='The number of threads for features extraction.')            # 特征提取的线程数量
feature_argparse.add_argument('--number_of_smali_files', type=int, default=1000000,
                              help='The maximum number of smali files to represent each app')   # 表示每个应用的smali文件的最大数量
feature_argparse.add_argument('--max_vocab_size', type=int, default=10000,
                              help='The maximum number of vocabulary size')                     # 词汇的最大数量
feature_argparse.add_argument('--update', action='store_true',
                              help='Whether update the existed features.')                      # 是否更新已存在的特征

# 定义与检测器相关的命令行参数
detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=True,
                               help='whether use cuda enable gpu or cpu.')                      # 是否使用CUDA启用GPU
#detector_argparse.add_argument('--seed', type=int, default=0,
                               #help='random seed.')                                             # 随机种子
detector_argparse.add_argument('--basic_dnn_name', type=str, default='20230724-230516',         # 20230920-125450 (Androzoo)
                               help='第二阶段的调用的 BASIC DNN model')                   
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='mini-batch size')                                          # mini-batch大小
detector_argparse.add_argument('--epochs', type=int, default=800,
                               help='number of epochs to train.')                               
detector_argparse.add_argument('--hidden_dim', type=int, default=512,
                               help='DAE hidden dim')                               
detector_argparse.add_argument('--dropout', type=float, default=0.2,
                               help='DAE dropout prob')                               
detector_argparse.add_argument('--lambda_reg', type=float, default=0.1,
                               help='lambda_reg')         
detector_argparse.add_argument('--learn_rate', type=float, default=0.001,
                               help='DAE learn rate')                         
detector_argparse.add_argument('--adv_eps', type=float, default=0.5,
                               help='adv_eps')      
detector_argparse.add_argument('--pro_eps', type=float, default=0.25,
                               help='pro_eps')         
detector_argparse.add_argument('--step', type=float, default=0.01,
                               help='step')        
detector_argparse.add_argument('--adv_depth', type=int, default=5,
                               help='adv_depth')       
detector_argparse.add_argument('--pro_depth', type=int, default=5,
                               help='pro_depth')        
detector_argparse.add_argument('--benign_count', type=int, default=3000,
                               help='benign_count')   
detector_argparse.add_argument('--malware_count', type=int, default=1000,
                               help='malware_count')   
detector_argparse.add_argument('--adversarial_count', type=int, default=2000,
                               help='adversarial_count')     
detector_argparse.add_argument('--add_noise_batch_size', type=int, default=128,
                               help='add_noise_batch_size')
detector_argparse.add_argument('--attention_dim', type=int, default=128,
                               help='attention_dim')                                   
detector_argparse.add_argument('--initial_noise_reduction_factor', type=float, default=0.1,
                               help='Initial noise figure ') # 初始噪声缩小系数     



# 定义与数据集相关的命令行参数
dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='use cache data or not.')                                   # 是否使用缓存数据

# 定义与模式相关的命令行参数
mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='learn a model or test it.')                                    # 学习模型或测试模型的模式
mode_argparse.add_argument('--dae_model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='suffix date of a tested model name.')                          # 测试模型名称的后缀日期
mode_argparse.add_argument('--dnn_model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='suffix date of a tested model name.')                          # 测试模型名称的后缀日期

# 定义主函数
def _main():
    # python -m examples.dae_test --cuda --cache --seed 0 --batch_size 128 --proc_number 10 --epochs 800 --max_vocab_size 10000 --hidden_dim 160  --learn_rate 0.001 --dropout 0.1 --basic_dnn_name "20230920-125450"
    args = cmd_md.parse_args()
    # 根据参数创建数据集
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    # 获取训练数据集输入生成器
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    # 获取验证数据集输入生成器
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
    # 获取测试数据集输入生成器
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    # 确保数据集的类别数为2
    assert dataset.n_classes == 2

    # 设置设备为CPU或CUDA
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'
    
    # 设置模型名称
    dae_model_name = args.dae_model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    dnn_model_name = args.dnn_model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")

    dae_model = malpurifier.dae(input_size = args.max_vocab_size,
                hidden_dim = args.hidden_dim,
                dropout_prob = args.dropout,
                lambda_reg = args.lambda_reg,
                device = dv,
                name = dae_model_name,
                predict_model_name = args.basic_dnn_name,
                adv_eps = args.adv_eps,
                pro_eps = args.pro_eps,
                step = args.step,
                adv_depth = args.adv_depth,
                pro_depth = args.pro_depth,
                benign_count = args.benign_count,
                malware_count = args.malware_count,
                adversarial_count = args.adversarial_count,
                add_noise_batch_size = 128,
                attention_dim = args.attention_dim,
                initial_noise_reduction_factor = args.initial_noise_reduction_factor
                )

    dnn_model = malpurifier.dnn(input_size=args.max_vocab_size,
                                n_classes=2,
                                device=dv,
                                name=dnn_model_name,
                                **vars(args)
                                )
    
    predict_model = MalwareDetectionDNN(input_size=args.max_vocab_size,
                                n_classes=2,
                                device=dv,
                                name=args.basic_dnn_name,
                                **vars(args)
                                )
    
    # 将模型移至指定设备并转换为双精度浮点数
    predict_model = predict_model.to(dv).double()

    predict_model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + args.basic_dnn_name,
                                         'model.pth')
    print("Basic DNN Model: ", predict_model_save_path)
    predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
    predict_model.eval()

    # 如果模式为训练，则进行模型拟合
    if args.mode == 'train':
        dae_model.to(dv)
        dnn_model.to(dv)
        malpurifier.fit(dae_model, dnn_model, predict_model, train_dataset_producer, val_dataset_producer, epochs=args.epochs, lr=args.learn_rate)
        # 将参数以人类可读的方式保存
        save_args(path.join(path.dirname(dae_model.model_save_path), "hparam"), vars(args))
        save_args(path.join(path.dirname(dnn_model.model_save_path), "hparam"), vars(args))
        # 将参数序列化，用于构建神经网络
        dump_pickle(vars(args), path.join(path.dirname(dae_model.model_save_path), "hparam.pkl"))
        dump_pickle(vars(args), path.join(path.dirname(dnn_model.model_save_path), "hparam.pkl"))
    else:
        dae_model.load()
        dae_model.to(dv)

        dnn_model.load()
        dnn_model.to(dv)
        malpurifier.predict(dae_model, dnn_model, test_dataset_producer)


if __name__ == '__main__':
    _main()
