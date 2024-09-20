from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, PGDAdvTraining, RFGSMAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE, VAE_SU
    
from torch.utils.data import DataLoader, TensorDataset
    
from autoattack import AutoAttack

from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.autoattack_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for autoattack')

atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='number of examples per batch.')

atta_argparse.add_argument('--norm', type=str, default='L2',
                           choices=['Linf', 'L2', 'L1'],
                           help="norm = ['Linf' | 'L2' | 'L1'] is the norm of the threat model")

atta_argparse.add_argument('--eps', type=float, default=1.,
                           help="eps is the bound on the norm of the adversarial perturbations")

atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus',
                                    'amd_pad_ma', 'fd_vae', 'dae'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae', 'dae'.")

atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')
atta_argparse.add_argument('--basic_dnn_name', type=str, default='20230724-230516',
                           help='basic_dnn_name')


def adapt_input(x):
    x = x.reshape(-1, 1, 1, 10000)
    return x

class AdaptedModel(nn.Module):
    def __init__(self, original_model):
        super(AdaptedModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x = adapt_input(x)
        x = x.view(x.size(0), -1)
        x = x.to(torch.float32)
        logits = self.original_model.forward(x)
        if isinstance(logits, tuple):
            return logits[0]
        else:
            return logits

def _main():
    args = atta_argparse.parse_args()
    
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'md_at_pgd':
        save_dir = config.get('experiments', 'md_at_pgd') + '_' + args.model_name
    elif args.model == 'md_at_ma':
        save_dir = config.get('experiments', 'md_at_ma') + '_' + args.model_name
    elif args.model == 'md_at_fgsm':
        save_dir = config.get('experiments', 'md_at_fgsm') + '_' + args.model_name
    elif args.model == 'amd_kde':
        save_dir = config.get('experiments', 'amd_kde') + '_' + args.model_name
    elif args.model == 'amd_icnn':
        save_dir = config.get('experiments', 'amd_icnn') + '_' + args.model_name
    elif args.model == 'amd_dla':
        save_dir = config.get('experiments', 'amd_dla') + '_' + args.model_name
    elif args.model == 'amd_dnn_plus':
        save_dir = config.get('experiments', 'amd_dnn_plus') + '_' + args.model_name
    elif args.model == 'amd_pad_ma':
        save_dir = config.get('experiments', 'amd_pad_ma') + '_' + args.model_name
    elif args.model == 'fd_vae':
        save_dir = config.get('experiments', 'fd_vae') + '_' + args.model_name
    elif args.model == 'dae':
        save_dir = config.get('experiments', 'dae') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae' and 'dae'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))

    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
    test_x, testy = dataset.test_dataset

    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)

    logger.info(f"Total number of malicious samples: {len(mal_test_x)}")
    
    mal_count = len(mal_testy)
    if mal_count <= 0:
        return

    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=args.batch_size,
                                                           name='test')

    assert dataset.n_classes == 2

    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
     
    from pprint import pprint   
    pprint(hp_params)
    pprint(args)
        
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    if args.model == 'amd_icnn' or args.model == 'amd_pad_ma':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       name=args.model_name,
                                       **hp_params
                                       )
    model = model.to(dv)

    if args.model == 'md_at_pgd':
        at_wrapper = PGDAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model
        
    elif args.model == 'md_at_ma':
        at_wrapper = MaxAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model
        
    elif args.model == 'md_at_fgsm':
        at_wrapper = RFGSMAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model  
                
    elif args.model == 'amd_kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()
        
    elif args.model == 'amd_dla':
        model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=args.model_name,
                                     **hp_params
                                     )
        model = model.to(dv)
        model.load()
        
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv)
        model.load()
        
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()
        model = adv_model.model

    elif args.model == 'fd_vae':
        model = VAE_SU(name=args.model_name)
        model.load()
        model.to(dv)

    elif args.model == 'dae':
        model = DAE(input_size = dataset.vocab_size,
            device = dv,
            name = args.model_name,
            predict_model_name = args.basic_dnn_name
            )
        model.load()
        model.to(dv)
        
        predict_model = MalwareDetectionDNN(dataset.vocab_size,
                            dataset.n_classes,
                            device=dv,
                            name=model.predict_model_name,
                            **hp_params
                            )
        predict_model = predict_model.to(dv)
        predict_model_save_path = os.path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        print("Basic DNN Model: ", predict_model_save_path)
        predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()
        
    else:
        model.load()

    logger.info("Load model parameters from {}.".format(model.model_save_path))

    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    
    x_adv_samples = []

    model.eval()
    
    adapted_model = AdaptedModel(model)
    
    adversary = AutoAttack(adapted_model, norm=args.norm, eps=args.eps, version='standard') 
    adversary.attacks_to_run = ['apgd-ce', 'fab'] 
    adversary.apgd.n_restarts = 2
    adversary.fab.n_restarts = 2

    if args.model == 'dae':
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.float(), y.long(), model.device)
            
            adapted_model = AdaptedModel(predict_model)
            
            adversary = AutoAttack(adapted_model, norm=args.norm, eps=args.eps, version='standard') 
            adversary.attacks_to_run = ['apgd-ce', 'fab'] 
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2
            adv_x_batch = adversary.run_standard_evaluation(x, y, args.batch_size) 

            adv_x_batch = adv_x_batch.to(torch.float32)

            Purified_adv_x_batch = model(adv_x_batch).to(torch.float32)

            Purified_adv_x_batch = Purified_adv_x_batch.to(model.device)
            
            y_cent_batch, _ = predict_model.inference_batch_wise(Purified_adv_x_batch)
            y_cent_list.append(y_cent_batch)

        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
        logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / len(y_pred) * 100:.3f}%')
    
    else:
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.float(), y.long(), model.device)

            adv_x_batch = adversary.run_standard_evaluation(x, y, args.batch_size) 

            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)

            y_cent_list.append(y_cent_batch)
            x_density_list.append(x_density_batch)
            x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
            x_adv_samples.append((adv_x_batch).detach().cpu().numpy())

        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)

        logger.info(
            f'The mean accuracy on autoattack is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

if __name__ == '__main__':
    _main()