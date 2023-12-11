from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, AdvMalwareDetectorICNN, AMalwareDetectionPAD
from core.attack import Max, PGD, PGDl1, StepwiseMax
from tools.utils import save_args, get_group_args, dump_pickle
from examples.amd_icnn_test import cmd_md

max_adv_argparse = cmd_md.add_argument_group(title='max adv training')
max_adv_argparse.add_argument('--beta_1', type=float, default=0.1, help='penalty factor on adversarial loss.')
max_adv_argparse.add_argument('--beta_2', type=float, default=1., help='penalty factor on adversary detector.')
max_adv_argparse.add_argument('--lambda_lb', type=float, default=1e-3,
                              help='the lower bound of penalty factor on adversary detector for looking for attacks.')
max_adv_argparse.add_argument('--lambda_ub', type=float, default=1e3,
                              help='the upper bound of penalty factor on adversary detector for looking for attacks.')
max_adv_argparse.add_argument('--detector', type=str, default='icnn',
                              choices=['none', 'icnn'],
                              help="detector type, either of 'icnn' and 'none'.")
max_adv_argparse.add_argument('--under_sampling', type=float, default=1.,
                              help='under-sampling ratio for adversarial training')
max_adv_argparse.add_argument('--ma', type=str, default='max', choices=['max', 'stepwise_max'],
                              help="Type of mixture of attack: 'max' or 'stepwise_max' strategy.")
max_adv_argparse.add_argument('--steps_l1', type=int, default=50,
                              help='maximum number of perturbations.')
max_adv_argparse.add_argument('--steps_l2', type=int, default=50,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                              help='step length in each step.')
max_adv_argparse.add_argument('--steps_linf', type=int, default=50,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length_linf', type=float, default=0.02,
                              help='step length in each step.')
max_adv_argparse.add_argument('--random_start', action='store_true', default=False,
                              help='randomly initialize the start points.')
max_adv_argparse.add_argument('--round_threshold', type=float, default=0.5,
                              help='threshold for rounding real scalars at the initialization step.')
max_adv_argparse.add_argument('--is_score_round', action='store_true', default=False,
                              help='whether scoring rule takes as input with rounding operation or not.')
max_adv_argparse.add_argument('--use_cont_pertb', action='store_true', default=False,
                              help='whether use the continuous perturbations for adversarial training.')


def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2 and args.epochs > 5

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=model_name,
                                **vars(args)
                                )
    if args.detector == 'icnn':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       name=model_name,
                                       **vars(args)
                                       )

    else:
        raise NotImplementedError
    model = model.to(dv).double()
    pgdlinf = PGD(norm='linf', use_random=False,
                  is_attacker=False,
                  device=model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.steps_linf,
                              step_length=args.step_length_linf,
                              verbose=False
                              )
    pgdl2 = PGD(norm='l2', use_random=False, is_attacker=False, device=model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.steps_l2,
                            step_length=args.step_length_l2,
                            verbose=False
                            )
    pgdl1 = PGDl1(is_attacker=False, device=model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            steps=args.steps_l1,
                            verbose=False)

    if args.ma == 'max':
        attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                     varepsilon=1e-9,
                     is_attacker=False,
                     device=model.device
                     )
        attack_param = {
            'steps_max': 1,  # steps for max attack
            'verbose': True
        }
    elif args.ma == 'stepwise_max':
        attack = StepwiseMax(is_attacker=False, device=model.device)
        attack_param = {
            'steps': max(max(args.steps_l1, args.steps_linf), args.steps_l2),
            'sl_l1': 1.,
            'sl_l2': args.step_length_l2,
            'sl_linf': args.step_length_linf,
            'is_score_round': args.is_score_round,
            'verbose': True
        }
    else:
        raise NotImplementedError("Expected 'max' and 'stepwise_max'.")
    
    # 初始化一个针对恶意软件检测的对抗性训练模型
    max_adv_training_model = AMalwareDetectionPAD(model, attack, attack_param)

    # 如果运行模式为“train”，则开始训练
    if args.mode == 'train':
        # 使用指定的参数进行对抗性训练
        max_adv_training_model.fit(
            train_dataset_producer,      # 训练数据生成器
            val_dataset_producer,        # 验证数据生成器
            adv_epochs=args.epochs,      # 对抗性训练的迭代次数
            beta_1=args.beta_1,          # 优化器Adam的beta1参数
            beta_2=args.beta_2,          # 优化器Adam的beta2参数
            lmda_lower_bound=args.lambda_lb,  # lambda的下界值
            lmda_upper_bound=args.lambda_ub,  # lambda的上界值
            use_continuous_pert=args.use_cont_pertb,  # 是否使用连续的扰动
            lr=args.lr,                  # 学习率
            under_sampling_ratio=args.under_sampling,  # 下采样比例
            weight_decay=args.weight_decay   # 权重衰减率，用于正则化
        )

        # 从硬盘加载模型
        max_adv_training_model.load()
        
        # 根据验证数据集计算决策阈值
        max_adv_training_model.model.get_threshold(val_dataset_producer)
        
        # 将模型保存到硬盘
        max_adv_training_model.save_to_disk(max_adv_training_model.model_save_path)

        # 以易读的格式保存模型的参数
        save_args(path.join(path.dirname(max_adv_training_model.model_save_path), "hparam"), vars(args))
        
        # 以pickle格式保存模型的参数，以便之后重建神经网络模型
        dump_pickle(vars(args), path.join(path.dirname(max_adv_training_model.model_save_path), "hparam.pkl"))

    # 加载模型并评估其准确性
    # 从硬盘加载模型
    max_adv_training_model.load()
    # 根据指定的比例和验证数据集计算决策阈值
    max_adv_training_model.model.get_threshold(val_dataset_producer, ratio=args.ratio)
    # 使用测试数据集评估模型的预测准确性
    max_adv_training_model.model.predict(test_dataset_producer, indicator_masking=True)



if __name__ == '__main__':
    _main()
