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
    args = cmd_md.parse_args()

    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))

    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(
        *dataset.test_dataset, batch_size=args.batch_size, name='test')

    assert dataset.n_classes == 2

    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime(
        "%Y%m%d-%H%M%S")

    dla_model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=model_name,
                                     **vars(args)
                                     )
    dla_model = dla_model.to(dv).double()

    pgdlinf = PGD(norm='linf', use_random=False,
                  is_attacker=False,
                  device=dla_model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.steps_linf,
                              step_length=args.step_length_linf,
                              verbose=False
                              )

    pgdl2 = PGD(norm='l2', use_random=False,
                is_attacker=False, device=dla_model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.steps_l2,
                            step_length=args.step_length_l2,
                            verbose=False
                            )

    pgdl1 = PGDl1(is_attacker=False, device=dla_model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            steps=args.steps_l1,
                            verbose=False)

    if args.ma == 'max':
        attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                     varepsilon=1e-9,
                     is_attacker=False,
                     device=dla_model.device
                     )
        attack_param = {
            'steps_max': 1,  # Max attack steps
            'verbose': True
        }

    elif args.ma == 'stepwise_max':
        attack = StepwiseMax(is_attacker=False, device=dla_model.device)
        attack_param = {
            'steps': max(max(args.steps_l1, args.steps_linf), args.steps_l2),
            'sl_l1': 1.,
            'sl_l2': args.step_length_l2,
            'sl_linf': args.step_length_linf,
            'verbose': True
        }
    else:
        raise NotImplementedError("Expected 'max' and 'stepwise_max'.")

    if args.mode == 'train':
        dla_model.fit(train_dataset_producer,
                      val_dataset_producer,
                      attack,
                      attack_param,
                      epochs=args.epochs,
                      lr=args.lr,
                      weight_decay=args.weight_decay
                      )
        save_args(path.join(path.dirname(
            dla_model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(
            dla_model.model_save_path), "hparam.pkl"))

    dla_model.load()
    dla_model.get_threshold(val_dataset_producer, ratio=args.ratio)
    dla_model.predict(test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main()
