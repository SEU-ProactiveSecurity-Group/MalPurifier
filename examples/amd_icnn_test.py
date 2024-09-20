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
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size,
                                                        name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size,
                                                      name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    dv = 'cuda' if args.cuda else 'cpu'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    md_model = MalwareDetectionDNN(dataset.vocab_size,
                                   dataset.n_classes,
                                   device=dv,
                                   name=model_name,
                                   **vars(args)
                                   )
    model = AdvMalwareDetectorICNN(md_model,
                                   input_size=dataset.vocab_size,
                                   n_classes=dataset.n_classes,
                                   device=dv,
                                   name=model_name,
                                   **vars(args)
                                   )
    model = model.to(dv).double()

    if args.mode == 'train':
        model.fit(train_dataset_producer,
                  val_dataset_producer,
                  epochs=args.epochs,
                  lr=args.lr,
                  weight_decay=args.weight_decay
                  )
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

        model.load()
        model.get_threshold(val_dataset_producer)
        
        model.save_to_disk()

        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.get_threshold(val_dataset_producer, ratio=args.ratio)
    model.predict(test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main()
