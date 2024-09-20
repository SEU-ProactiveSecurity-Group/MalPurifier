# Ensure consistent behavior in Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import argparse
import time

from core.defense import Dataset
from core.defense import MalwareDetectionLSTM
from tools.utils import save_args, get_group_args, dump_pickle

# Initialize argparse object for parsing command line arguments
cmd_md = argparse.ArgumentParser(description='Arguments for malware detection using LSTM')

# Define command line arguments for feature extraction
feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=10,
                              help='Number of threads for feature extraction')
feature_argparse.add_argument('--number_of_smali_files', type=int, default=1000000,
                              help='Maximum number of smali files to represent each app')
feature_argparse.add_argument('--max_vocab_size', type=int, default=10000,
                              help='Maximum vocabulary size')
feature_argparse.add_argument('--update', action='store_true',
                              help='Update existing features')

# Define command line arguments for the detector
detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=False,
                               help='Use CUDA for GPU acceleration')
detector_argparse.add_argument('--seed', type=int, default=0,
                               help='Random seed for reproducibility')
detector_argparse.add_argument('--hidden_dim', type=int, default=200,
                               help='Dimension of LSTM hidden state')
detector_argparse.add_argument('--seq_len', type=int, default=1,
                               help='Sequence length')
detector_argparse.add_argument('--dropout', type=float, default=0.6,
                               help='Dropout rate')
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='Mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=50,
                               help='Number of training epochs')
detector_argparse.add_argument('--lr', type=float, default=0.001,
                               help='Initial learning rate')
detector_argparse.add_argument('--weight_decay', type=float, default=0,
                               help='Weight decay coefficient')

# Define command line arguments for the dataset
dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='Use cached data')

# Define command line arguments for the mode
mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                           help='Train a model or test it')
mode_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx',
                           help='Suffix date of a tested model name')

def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    dv = 'cuda' if args.cuda else 'cpu'
    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    model = MalwareDetectionLSTM(input_dim=dataset.vocab_size//args.seq_len,
                                n_classes=dataset.n_classes,
                                device=dv,
                                name=model_name,
                                **vars(args))
    model = model.to(dv).double()

    if args.mode == 'train':
        model.fit(train_dataset_producer, val_dataset_producer, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.predict(test_dataset_producer)

if __name__ == '__main__':
    _main()
