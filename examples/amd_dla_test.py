from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial

from core.defense import Dataset
from core.defense import AMalwareDetectionDLA
from core.attack import Max, PGD, PGDl1, StepwiseMax
from tools.utils import save_args, get_group_args, dump_pickle
from examples.amd_icnn_test import cmd_md

dla_argparse = cmd_md.add_argument_group(title='amd dla')
dla_argparse.add_argument('--ma', type=str, default='max', choices=['max', 'stepwise_max'],
                          help="Type of mixture of attack: 'max' or 'stepwise_max' strategy.")
dla_argparse.add_argument('--steps_l1', type=int, default=50,
                          help='maximum number of perturbations.')
dla_argparse.add_argument('--steps_l2', type=int, default=50,
                          help='maximum number of steps for base attacks.')
dla_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                          help='step length in each step.')
dla_argparse.add_argument('--steps_linf', type=int, default=50,
                          help='maximum number of steps for base attacks.')
dla_argparse.add_argument('--step_length_linf', type=float, default=0.02,
                          help='step length in each step.')
dla_argparse.add_argument('--random_start', action='store_true', default=False,
                          help='randomly initialize the start points.')
dla_argparse.add_argument('--round_threshold', type=float, default=0.5,
                          help='threshold for rounding real scalars at the initialization step.')


def _main():
    # 解析命令行参数
    args = cmd_md.parse_args()
    
    # 根据提供的参数初始化数据集
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    
    # 为训练、验证和测试数据集建立输入生产器
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    
    # 断言确保数据集只有两个类别
    assert dataset.n_classes == 2

    # 检查是否使用cuda加速，若不使用则选择cpu作为设备
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    # 为模型设置名称，如果是测试模式则使用给定的名称，否则使用当前时间为其命名
    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    # 初始化深度学习对抗模型
    dla_model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=model_name,
                                     **vars(args)
                                     )
    # 将模型移到相应的设备上并将其转化为双精度格式
    dla_model = dla_model.to(dv).double()
    
    # 初始化一个基于L-infinity范数的PGD对抗攻击器
    pgdlinf = PGD(norm='linf', use_random=False,
                  is_attacker=False,
                  device=dla_model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.steps_linf,
                              step_length=args.step_length_linf,
                              verbose=False
                              )
    
    # 初始化一个基于L2范数的PGD对抗攻击器
    pgdl2 = PGD(norm='l2', use_random=False, is_attacker=False, device=dla_model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.steps_l2,
                            step_length=args.step_length_l2,
                            verbose=False
                            )
    
    # 初始化一个基于L1范数的PGD对抗攻击器
    pgdl1 = PGDl1(is_attacker=False, device=dla_model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            steps=args.steps_l1,
                            verbose=False)

    # 根据用户参数选择攻击策略
    if args.ma == 'max':
        # Max攻击结合了三种PGD攻击方法
        attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                    varepsilon=1e-9,
                    is_attacker=False,
                    device=dla_model.device
                    )
        # 为Max攻击设置参数
        attack_param = {
            'steps_max': 1,  # Max攻击的步骤数
            'verbose': True
        }

    elif args.ma == 'stepwise_max':
        # StepwiseMax是另一种攻击策略
        attack = StepwiseMax(is_attacker=False, device=dla_model.device)
        # 设置StepwiseMax攻击的参数
        attack_param = {
            'steps': max(max(args.steps_l1, args.steps_linf), args.steps_l2), #选择三种范数中的最大步数
            'sl_l1': 1.,
            'sl_l2': args.step_length_l2,
            'sl_linf': args.step_length_linf,
            'verbose': True
        }
    else:
        # 如果用户的选择不在上述两者之内，则抛出未实现的错误
        raise NotImplementedError("Expected 'max' and 'stepwise_max'.")

    # 如果是训练模式
    if args.mode == 'train':
        dla_model.fit(train_dataset_producer,
                    val_dataset_producer,
                    attack,
                    attack_param,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay
                    )
        # 保存参数为人类可读格式
        save_args(path.join(path.dirname(dla_model.model_save_path), "hparam"), vars(args))
        # 为了重建神经网络保存参数
        dump_pickle(vars(args), path.join(path.dirname(dla_model.model_save_path), "hparam.pkl"))

    # 加载模型
    dla_model.load()
    # 根据验证集获取阈值
    dla_model.get_threshold(val_dataset_producer, ratio=args.ratio)
    # 对测试数据进行预测
    dla_model.predict(test_dataset_producer, indicator_masking=True)

if __name__ == '__main__':
    _main()