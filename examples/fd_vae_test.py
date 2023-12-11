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
from core.defense import VAE_SU
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
detector_argparse.add_argument('--seed', type=int, default=0,
                               help='random seed.')                                             # 随机种子                  
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='mini-batch size')                                          # mini-batch大小
detector_argparse.add_argument('--epochs', type=int, default=20,
                               help='number of epochs to train.')                               
detector_argparse.add_argument('--hidden_dim', type=int, default=200,
                               help='VAE_SU hidden dim')         
detector_argparse.add_argument('--dim_z', type=int, default=20,
                               help='dim_z')                                                              
detector_argparse.add_argument('--learn_rate', type=float, default=0.001,
                               help='learn rate')                         
detector_argparse.add_argument('--KLW', type=int, default=10,
                               help='KLW')                         
detector_argparse.add_argument('--NLOSSW', type=int, default=10,
                               help='NLOSSW')       
detector_argparse.add_argument('--metric', type=int, default=400,
                               help='metric')             


# 定义与数据集相关的命令行参数
dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='use cache data or not.')                                   # 是否使用缓存数据

# 定义与模式相关的命令行参数
mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='learn a model or test it.')                                    # 学习模型或测试模型的模式
mode_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='suffix date of a tested model name.')                          # 测试模型名称的后缀日期

# 定义主函数
def _main():
    # python -m examples.dae_test --cuda --cache --seed 0 --batch_size 128 --proc_number 10 --epochs 800 --max_vocab_size 10000 --hidden_dim 160  --learn_rate 0.001 --dropout 0.1 --basic_dnn_name "20230920-125450"
    args = cmd_md.parse_args()
    # 根据参数创建数据集
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    # 获取训练数据集输入生成器
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    # # 获取验证数据集输入生成器
    # val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
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
    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    model = VAE_SU(input_size=args.max_vocab_size,
                   n_hidden=args.hidden_dim,
                   n_epochs=args.epochs,
                   z_dim=args.dim_z,
                   learn_rate=args.learn_rate,
                   Loss_type="1",
                   KLW=args.KLW,
                   NLOSSW=args.NLOSSW,
                   name=model_name,
                   device=dv,
                   )
    
    # model = VAE_SU(name=model_name)
    
    # 将模型移至指定设备并转换为双精度浮点数
    model = model.to(dv)

    # 如果模式为训练，则进行模型拟合
    if args.mode == 'train':
        model.to(dv)
        model.fit(train_dataset_producer)
        # 将参数以人类可读的方式保存
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        # 将参数序列化，用于构建神经网络
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    # # 加载模型并进行预测
    model.load()
    model.to(dv)
    model.predict(test_dataset_producer, indicator_masking=True, metric=args.metric)

if __name__ == '__main__':
    _main()
