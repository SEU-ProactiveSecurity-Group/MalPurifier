from __future__ import absolute_import, division, print_function

import os.path as path
import argparse
import time

from core.defense import Dataset, MalwareDetectionRNN
from tools.utils import save_args, get_group_args, dump_pickle

cmd_md = argparse.ArgumentParser(description='Arguments for malware detector')

feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=2,
                              help='Number of threads for feature extraction.')
feature_argparse.add_argument('--number_of_smali_files', type=int, default=1000000,
                              help='Maximum number of smali files to represent each app')
feature_argparse.add_argument('--max_vocab_size', type=int, default=10000,
                              help='Maximum vocabulary size')
feature_argparse.add_argument('--update', action='store_true',
                              help='Update existing features')

detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=True,
                               help='Use CUDA-enabled GPU')
detector_argparse.add_argument('--seed', type=int, default=0,
                               help='Random seed')
detector_argparse.add_argument('--hidden_size', type=int, default=200,
                               help='Size of RNN hidden state')
detector_argparse.add_argument('--num_layers', type=int, default=3,
                               help='Number of RNN layers')
detector_argparse.add_argument('--dropout', type=float, default=0.6,
                               help='Dropout rate')
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='Mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=50,
                               help='Number of epochs to train')
detector_argparse.add_argument('--lr', type=float, default=0.001,
                               help='Initial learning rate')
detector_argparse.add_argument('--weight_decay', type=float, default=0e-4,
                               help='Weight decay coefficient')

dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='Use cached data')

mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='Train or test the model')
mode_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='Suffix date of tested model name')

def main():
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    device = 'cuda' if args.cuda else 'cpu'
    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")

    model = MalwareDetectionRNN(input_size=dataset.vocab_size,
                                n_classes=dataset.n_classes,
                                device=device, 
                                name=model_name, 
                                **vars(args))

    model = model.to(device).double()

    if args.mode == 'train':
        model.fit(train_dataset_producer, val_dataset_producer, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.predict(test_dataset_producer)

if __name__ == '__main__':
    main()
