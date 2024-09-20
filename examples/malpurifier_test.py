# Ensure consistent behavior in Python 2 and 3
from __future__ import absolute_import, division, print_function

# Import required libraries
import os.path as path
import argparse
import time
import torch

# Import custom modules
from core.defense import Dataset, malpurifier, MalwareDetectionDNN
from tools.utils import save_args, get_group_args, dump_pickle
from config import config

# Initialize argparse object for parsing command line arguments
cmd_md = argparse.ArgumentParser(description='Arguments for malware detector')

# Define command line arguments for feature extraction
feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=2,
                              help='Number of threads for feature extraction.')
feature_argparse.add_argument('--number_of_smali_files', type=int, default=1000000,
                              help='Maximum number of smali files to represent each app')
feature_argparse.add_argument('--max_vocab_size', type=int, default=10000,
                              help='Maximum vocabulary size')
feature_argparse.add_argument('--update', action='store_true',
                              help='Update existing features')

# Define command line arguments for detector
detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=True,
                               help='Use CUDA-enabled GPU')
detector_argparse.add_argument('--basic_dnn_name', type=str, default='20230724-230516',
                               help='Basic DNN model name for second stage')
detector_argparse.add_argument('--batch_size', type=int, default=128,
                               help='Mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=800,
                               help='Number of epochs to train')
detector_argparse.add_argument('--hidden_dim', type=int, default=512,
                               help='DAE hidden dimension')
detector_argparse.add_argument('--dropout', type=float, default=0.2,
                               help='DAE dropout probability')
detector_argparse.add_argument('--lambda_reg', type=float, default=0.1,
                               help='Lambda regularization')
detector_argparse.add_argument('--learn_rate', type=float, default=0.001,
                               help='DAE learning rate')
detector_argparse.add_argument('--adv_eps', type=float, default=0.5,
                               help='Adversarial epsilon')
detector_argparse.add_argument('--pro_eps', type=float, default=0.25,
                               help='Projection epsilon')
detector_argparse.add_argument('--step', type=float, default=0.01,
                               help='Step size')
detector_argparse.add_argument('--adv_depth', type=int, default=5,
                               help='Adversarial depth')
detector_argparse.add_argument('--pro_depth', type=int, default=5,
                               help='Projection depth')
detector_argparse.add_argument('--benign_count', type=int, default=3000,
                               help='Number of benign samples')
detector_argparse.add_argument('--malware_count', type=int, default=1000,
                               help='Number of malware samples')
detector_argparse.add_argument('--adversarial_count', type=int, default=2000,
                               help='Number of adversarial samples')
detector_argparse.add_argument('--add_noise_batch_size', type=int, default=128,
                               help='Batch size for adding noise')
detector_argparse.add_argument('--attention_dim', type=int, default=128,
                               help='Attention dimension')
detector_argparse.add_argument('--initial_noise_reduction_factor', type=float, default=0.1,
                               help='Initial noise reduction factor')

# Define command line arguments for dataset
dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--cache', action='store_true', default=False,
                               help='Use cached data')

# Define command line arguments for mode
mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='Train or test the model')
mode_argparse.add_argument('--dae_model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='Suffix date of tested DAE model name')
mode_argparse.add_argument('--dnn_model_name', type=str, default='xxxxxxxx-xxxxxx', required=False,
                           help='Suffix date of tested DNN model name')

def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(feature_ext_args=get_group_args(args, cmd_md, 'feature'))
    train_dataset_producer = dataset.get_input_producer(*dataset.train_dataset, batch_size=args.batch_size, name='train', use_cache=args.cache)
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(*dataset.test_dataset, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    dv = 'cuda' if args.cuda else 'cpu'
    
    dae_model_name = args.dae_model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    dnn_model_name = args.dnn_model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")

    dae_model = malpurifier.dae(input_size=args.max_vocab_size,
                hidden_dim=args.hidden_dim,
                dropout_prob=args.dropout,
                lambda_reg=args.lambda_reg,
                device=dv,
                name=dae_model_name,
                predict_model_name=args.basic_dnn_name,
                adv_eps=args.adv_eps,
                pro_eps=args.pro_eps,
                step=args.step,
                adv_depth=args.adv_depth,
                pro_depth=args.pro_depth,
                benign_count=args.benign_count,
                malware_count=args.malware_count,
                adversarial_count=args.adversarial_count,
                add_noise_batch_size=args.add_noise_batch_size,
                attention_dim=args.attention_dim,
                initial_noise_reduction_factor=args.initial_noise_reduction_factor
                )

    dnn_model = malpurifier.dnn(input_size=args.max_vocab_size,
                                n_classes=2,
                                device=dv,
                                name=dnn_model_name,
                                **vars(args)
                                )
    
    predict_model = MalwareDetectionDNN(input_size=args.max_vocab_size,
                                n_classes=2,
                                device=dv,
                                name=args.basic_dnn_name,
                                **vars(args)
                                )
    
    predict_model = predict_model.to(dv).double()

    predict_model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + args.basic_dnn_name,
                                         'model.pth')
    print("Basic DNN Model: ", predict_model_save_path)
    predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
    predict_model.eval()

    if args.mode == 'train':
        dae_model.to(dv)
        dnn_model.to(dv)
        malpurifier.fit(dae_model, dnn_model, predict_model, train_dataset_producer, val_dataset_producer, epochs=args.epochs, lr=args.learn_rate)
        save_args(path.join(path.dirname(dae_model.model_save_path), "hparam"), vars(args))
        save_args(path.join(path.dirname(dnn_model.model_save_path), "hparam"), vars(args))
        dump_pickle(vars(args), path.join(path.dirname(dae_model.model_save_path), "hparam.pkl"))
        dump_pickle(vars(args), path.join(path.dirname(dnn_model.model_save_path), "hparam.pkl"))
    else:
        dae_model.load()
        dae_model.to(dv)
        dnn_model.load()
        dnn_model.to(dv)
        malpurifier.predict(dae_model, dnn_model, test_dataset_producer)

if __name__ == '__main__':
    _main()
