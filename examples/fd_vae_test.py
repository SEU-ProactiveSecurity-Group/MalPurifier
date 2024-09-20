from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import argparse
import time
import numpy
import torch

from core.defense import Dataset
from core.defense import VAE_SU
from tools.utils import save_args, get_group_args, to_tensor, dump_pickle, read_pickle

from config import config, logging, ErrorHandler

cmd_md = argparse.ArgumentParser(description='arguments for learning malware detector')

feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=2,
                              help='The number of threads for features extraction.')
feature_argparse.add_argument('--number_of_smali_files', type=int, default=1000000,
                              help='The maximum number of smali files to represent each app')
feature_argparse.add_argument('--max_vocab_size', type=int, default=10000,
                              help='The maximum number of vocabulary size')
feature_argparse.add_argument('--update', action='store_true',
                              help='Whether update the existed features.')

detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=True,
                               help='whether use cuda enable gpu or cpu.')
detector_argparse.add_argument('--seed', type=int, default=0,
                               help='random seed.')                  
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=20,
                               help='number of epochs to train.')                               
detector_argparse.add_argument('--hidden_dim', type=int, default=200,
                               help='VAE_SU hidden dim')         
detector_argparse.add_argument('--dim_z', type=int, default=20,
                               help='dim_z')                                                              
detector_argparse.add_argument('--learn_rate', type=float, default=0.001,
                               help='learn rate')                         
detector_argparse.add_argument('--KLW', type=int, default=10,
                               help='KLW')                         
detector_argparse.add_argument('--NLOSSW', type=int, default=10,
                               help='NLOSSW')       
detector_argparse.add_argument('--metric', type=int, default=400,
                               help='metric')             

dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='use cache data or not.')

mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='learn a model or test it.')
mode_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='suffix date of a tested model name.')

def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    dv = 'cuda' if args.cuda else 'cpu'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    model = VAE_SU(input_size=args.max_vocab_size,
                   n_hidden=args.hidden_dim,
                   n_epochs=args.epochs,
                   z_dim=args.dim_z,
                   learn_rate=args.learn_rate,
                   Loss_type="1",
                   KLW=args.KLW,
                   NLOSSW=args.NLOSSW,
                   name=model_name,
                   device=dv,
                   )
    
    model = model.to(dv)

    if args.mode == 'train':
        model.to(dv)
        model.fit(train_dataset_producer)
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.to(dv)
    model.predict(test_dataset_producer, indicator_masking=True, metric=args.metric)

if __name__ == '__main__':
    _main()
