from __future__ import absolute_import, division, print_function

import os.path as path
import argparse
import time

from core.defense import Dataset, MalwareDetectionRF
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
detector_argparse.add_argument('--seed', type=int, default=0, help='Random seed')
detector_argparse.add_argument('--n_estimators', type=int, default=500, help='Number of trees')
detector_argparse.add_argument('--max_depth', type=int, default=None, help='Maximum tree depth')
detector_argparse.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')

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
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    model = MalwareDetectionRF(n_estimators=args.n_estimators,
                               max_depth=args.max_depth,
                               random_state=args.seed,
                               name=model_name,
                               device='cpu')

    if args.mode == 'train':
        model.fit(train_dataset_producer)
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.predict(test_dataset_producer)

if __name__ == '__main__':
    main()
