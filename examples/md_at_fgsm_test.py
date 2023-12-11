from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import time
from functools import partial

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, RFGSMAdvTraining
from core.attack import RFGSM
from tools.utils import save_args, get_group_args, dump_pickle
from examples.amd_icnn_test import cmd_md

max_adv_argparse = cmd_md.add_argument_group(title='max adv training')
max_adv_argparse.add_argument('--beta', type=float, default=0.1, help='penalty factor on adversarial loss.')
max_adv_argparse.add_argument('--steps', type=int, default=5,
                              help='maximum number of steps for base attacks.')
max_adv_argparse.add_argument('--step_length', type=float, default=0.02,
                              help='step length in each step.')

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
    model = model.to(dv).double()

    attack = RFGSM(is_attacker=False,
                   random=True,
                   oblivion=False,
                   kappa=1,
                   device=model.device)
    
    attack.perturb = partial(attack.perturb,
                             steps=args.steps,
                             step_length=args.step_length,
                             verbose=True
                             )
    attack_param = {}
    max_adv_training_model = RFGSMAdvTraining(model, attack, attack_param)
    
    if args.mode == 'train':
        max_adv_training_model.fit(train_dataset_producer,
                                   val_dataset_producer,
                                   epochs=5,
                                   adv_epochs=args.epochs - 5,
                                   beta=args.beta,
                                   lr=args.lr,
                                   weight_decay=args.weight_decay
                                   )
        
        # human readable parameters
        save_args(path.join(path.dirname(max_adv_training_model.model_save_path), "hparam"), vars(args))
        # save parameters for rebuilding the neural nets
        dump_pickle(vars(args), path.join(path.dirname(max_adv_training_model.model_save_path), "hparam.pkl"))
        
    # test: accuracy
    max_adv_training_model.load()
    max_adv_training_model.model.predict(test_dataset_producer)


if __name__ == '__main__':
    _main()
