from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time

from core.defense import Dataset
from core.defense import AdvMalwareDetectorICNN, MalwareDetectionDNN
from tools.utils import save_args, get_group_args, dump_pickle
from examples.md_nn_test import cmd_md

indicator_argparse = cmd_md.add_argument_group(title='adv indicator')
indicator_argparse.add_argument('--ratio', type=float, default=0.95,
                                help='ratio of validation examples remained for passing through malware detector')


def _main():
    # 解析命令行参数
    args = cmd_md.parse_args()
    # 加载数据集，并根据参数提取特征
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    # 获取训练数据集的输入生成器
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train', use_cache=args.cache)
    # 获取验证数据集的输入生成器
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    # 获取测试数据集的输入生成器
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    # 确保数据集只有两个类别（可能是恶意软件和非恶意软件）
    assert dataset.n_classes == 2

    # 根据是否使用CUDA选择设备（CPU或GPU）
    dv = 'cuda' if args.cuda else 'cpu'

    # 如果是测试模式，则使用给定的模型名称，否则使用当前时间生成一个模型名称
    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    # 初始化基础的恶意软件检测模型
    md_model = MalwareDetectionDNN(dataset.vocab_size,
                                   dataset.n_classes,
                                   device=dv,
                                   name=model_name,
                                   **vars(args)
                                   )
    # 初始化对抗恶意软件检测模型
    model = AdvMalwareDetectorICNN(md_model,
                                   input_size=dataset.vocab_size,
                                   n_classes=dataset.n_classes,
                                   device=dv,
                                   name=model_name,
                                   **vars(args)
                                   )
    # 将模型转移到指定设备并将其数据类型设置为double
    model = model.to(dv).double()

    # 如果处于训练模式
    if args.mode == 'train':
        # 训练模型
        model.fit(train_dataset_producer,
                  val_dataset_producer,
                  epochs=args.epochs,
                  lr=args.lr,
                  weight_decay=args.weight_decay
                  )
        # 保存模型参数为人类可读格式
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        # 保存模型参数为pickle格式，方便后续重建神经网络
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

        # 加载模型并获取验证集的阈值
        model.load()
        model.get_threshold(val_dataset_producer)
        
        # 将模型保存到磁盘
        model.save_to_disk()

        # 再次保存模型参数为人类可读格式和pickle格式
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    # 加载模型并根据给定的比率获取阈值
    model.load()
    model.get_threshold(val_dataset_producer, ratio=args.ratio)
    # 测试模型的准确性
    model.predict(test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main()
