# Import future features for consistent behavior in Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import argparse
import time
import numpy
import torch

from core.defense import Dataset
from core.defense import DAE
from core.defense import MalwareDetectionDNN
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
detector_argparse.add_argument('--basic_dnn_name', type=str, default='20231016-121617',         
                               help='Basic DNN model for second stage')                   
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=400,
                               help='number of epochs to train.')                               
detector_argparse.add_argument('--hidden_dim', type=int, default=600,
                               help='DAE hidden dim')                               
detector_argparse.add_argument('--dropout', type=float, default=0.2,
                               help='DAE dropout prob')                               
detector_argparse.add_argument('--lambda_reg', type=float, default=0.001,
                               help='lambda_reg')         
detector_argparse.add_argument('--learn_rate', type=float, default=0.001,
                               help='DAE learn rate')                         
detector_argparse.add_argument('--adv_eps', type=float, default=0.5,
                               help='adv_eps')      
detector_argparse.add_argument('--pro_eps', type=float, default=0.25,
                               help='pro_eps')         
detector_argparse.add_argument('--step', type=float, default=0.01,
                               help='step')        
detector_argparse.add_argument('--adv_depth', type=int, default=5,
                               help='adv_depth')       
detector_argparse.add_argument('--pro_depth', type=int, default=5,
                               help='pro_depth')        
detector_argparse.add_argument('--benign_count', type=int, default=3000,
                               help='benign_count')   
detector_argparse.add_argument('--malware_count', type=int, default=1000,
                               help='malware_count')   
detector_argparse.add_argument('--adversarial_count', type=int, default=2000,
                               help='adversarial_count')     
detector_argparse.add_argument('--add_noise_batch_size', type=int, default=128,
                               help='add_noise_batch_size')
detector_argparse.add_argument('--attention_dim', type=int, default=200,
                               help='attention_dim')                                   
detector_argparse.add_argument('--initial_noise_reduction_factor', type=float, default=0.01,
                               help='Initial noise figure ')

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
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    dv = 'cuda' if args.cuda else 'cpu'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    
    model = DAE(input_size = args.max_vocab_size,
                hidden_dim = args.hidden_dim,
                dropout_prob = args.dropout,
                lambda_reg = args.lambda_reg,
                device = dv,
                name = model_name,
                predict_model_name = args.basic_dnn_name,
                adv_eps = args.adv_eps,
                pro_eps = args.pro_eps,
                step = args.step,
                adv_depth = args.adv_depth,
                pro_depth = args.pro_depth,
                benign_count = args.benign_count,
                malware_count = args.malware_count,
                adversarial_count = args.adversarial_count,
                add_noise_batch_size = 128,
                attention_dim = args.attention_dim,
                initial_noise_reduction_factor = args.initial_noise_reduction_factor
                )
    
    predict_model = MalwareDetectionDNN(input_size=args.max_vocab_size,
                                n_classes=2,
                                device=dv,
                                name=model_name,
                                **vars(args)
                                )
    
    predict_model = predict_model.to(dv).double()

    predict_model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + args.basic_dnn_name,
                                         'model.pth')
    print("Basic DNN Model: ", predict_model_save_path)
    predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
    predict_model.eval()

    if args.mode == 'train':
        model.to(dv)
        model.fit(predict_model, train_dataset_producer, test_dataset_producer, epochs=args.epochs, lr=args.learn_rate)
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    model.load()
    model.to(dv)
    model.predict(test_dataset_producer, predict_model)

if __name__ == '__main__':
    _main()
