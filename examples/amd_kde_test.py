from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, KernelDensityEstimation
from tools.utils import save_args, dump_pickle
from config import config
from tools import utils

import argparse

# 初始化一个 ArgumentParser 对象，用于解析命令行参数。描述为“核密度估计的参数”
kde_argparse = argparse.ArgumentParser(description='arguments for kernel density estimation')

# 添加一个命令行参数 --n_centers，类型为整数， 默认值为500，描述为“分布的数量”
kde_argparse.add_argument('--n_centers', type=int, default=500, help='number of distributions')

# 添加一个命令行参数 --bandwidth，类型为浮点数， 默认值为20.，描述为“高斯核的方差”
kde_argparse.add_argument('--bandwidth', type=float, default=20., help='variance of Gaussian kernel')

# 添加一个命令行参数 --ratio，类型为浮点数， 默认值为0.95，描述为“保留的验证示例的百分比”
kde_argparse.add_argument('--ratio', type=float, default=0.9, help='the percentage of reminded validation examples')

# 添加一个命令行参数 --cache，它是一个标志， 默认值为False，描述为“是否使用缓存数据”
kde_argparse.add_argument('--cache', action='store_true', default=False, help='use cache data or not.')

# 添加一个命令行参数 --mode，类型为字符串， 默认值为'train'，可选值为 ['train', 'test']，描述为“学习模型或测试模型”
kde_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False, help='learn a model or test it.')

# 添加一个命令行参数 --model_name，类型为字符串， 默认值为'xxxxxxxx-xxxxxx'，描述为“模型的时间戳”
kde_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')

def _main():
    # 解析命令行参数
    args = kde_argparse.parse_args()
    # 根据模型名称和配置文件确定保存的目录
    save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    # 从保存目录中读取模型超参数
    hp_params = utils.read_pickle(path.join(save_dir, 'hparam.pkl'))
    # 根据超参数中的proc_number加载数据集
    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
    # 获取训练数据集的输入生成器
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=hp_params['batch_size'],
                                                        name='train', use_cache=args.cache)
    # 获取验证数据集的输入生成器
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=hp_params['batch_size'],
                                                      name='val')
    # 获取测试数据集的输入生成器
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=hp_params['batch_size'],
                                                       name='test')
    # 确保数据集只有两个类别（可能是恶意软件和非恶意软件）
    assert dataset.n_classes == 2

    # 根据超参数决定使用CPU还是CUDA
    dv = 'cuda' if hp_params['cuda'] else 'cpu'
    
    # 初始化恶意软件检测的深度神经网络模型
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    # 将模型转移到指定设备并转换其数据类型为double
    model = model.to(dv).double()
    # 从磁盘加载模型
    model.load()
    
    # 使用Kernel Density Estimation (KDE) 方法对模型进行进一步的封装
    kde = KernelDensityEstimation(model,
                                  n_centers=args.n_centers,
                                  bandwidth=args.bandwidth,
                                  n_classes=dataset.n_classes,
                                  ratio=args.ratio
                                  )

    # 如果是在训练模式
    if args.mode == 'train':
        # 训练KDE模型
        kde.fit(train_dataset_producer, val_dataset_producer)
        
        # 将模型的参数以人类可读格式保存
        save_args(path.join(path.dirname(kde.model_save_path), "hparam"), {**vars(args), **hp_params})
        # 将模型的参数序列化为pickle格式并保存
        dump_pickle({**vars(args), **hp_params}, path.join(path.dirname(kde.model_save_path), "hparam.pkl"))
        # 将KDE模型保存到磁盘
        kde.save_to_disk()

    # 从磁盘加载KDE模型
    kde.load()
    # 获取验证数据集上的阈值
    kde.get_threshold(val_dataset_producer, ratio=args.ratio)
    # 使用KDE模型进行预测并评估其在测试集上的性能
    kde.predict(test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main()
