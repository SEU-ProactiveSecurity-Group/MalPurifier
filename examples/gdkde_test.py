"""
Warning: a large number of samples belonging to negative class will trigger the MOE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import argparse

import numpy as np
import torch

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, PGDAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE
    
from core.attack import GDKDE
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.gdkde_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for l1 norm based projected gradient descent attack')
atta_argparse.add_argument('--steps', type=int, default=50,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length', type=float, default=2.,
                           help='step length in each step.')
atta_argparse.add_argument('--bandwidth', type=float, default=20.,
                           help='variance of Gaussian distribution.')
atta_argparse.add_argument('--penalty_factor', type=float, default=1000.,
                           help='penalty factor for density estimation.')
atta_argparse.add_argument('--n_center', type=int, default=1000,
                           help='number of centers.')
atta_argparse.add_argument('--base', type=float, default=10.,
                           help='base of a logarithm function.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='attack confidence.')
atta_argparse.add_argument('--n_sample_times', type=int, default=1,
                           help='data sampling times when waging attacks')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='number of examples loaded in per batch.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'dae'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'dae'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')
atta_argparse.add_argument('--basic_dnn_name', type=str, default='20230724-230516',
                           help='basic_dnn_name')

def _main():
    args = atta_argparse.parse_args()
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'md_at_pgd':
        save_dir = config.get('experiments', 'md_at_pgd') + '_' + args.model_name
    elif args.model == 'md_at_ma':
        save_dir = config.get('experiments', 'md_at_ma') + '_' + args.model_name
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
    elif args.model == 'dae':
        save_dir = config.get('experiments', 'dae') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma' and 'dae'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
    test_x, testy = dataset.test_dataset
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)
    mal_count = len(mal_testy)

    # 打印出总共恶意样本的数量
    logger.info(f"⭐Total number of malicious samples: {len(mal_test_x)}")
    

    ben_test_x, ben_testy = test_x[testy == 0], testy[testy == 0]
    ben_count = len(ben_test_x)
    if mal_count <= 0 and ben_count <= 0:
        return
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=args.batch_size,
                                                           name='test')
    ben_test_dataset_producer = dataset.get_input_producer(ben_test_x, ben_testy,
                                                           batch_size=args.batch_size,
                                                           name='test'
                                                           )

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
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
    model = model.to(dv).double()
    if args.model == 'md_at_pgd':
        at_wrapper = PGDAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model
    elif args.model == 'md_at_ma':
        at_wrapper = MaxAdvTraining(model)
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
        model = model.to(dv).double()
        model.load()
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv).double()
        model.load()
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()
        model = adv_model.model
        
    elif args.model == 'dae':
        model = DAE(input_size = dataset.vocab_size,
            hidden_dim = 160,
            dropout_prob = 0.1,
            lambda_reg = 0.1,
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
        predict_model = predict_model.to(dv).double()
        import os.path as path
        predict_model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        print("Basic DNN Mode: ", predict_model_save_path)
        predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()
                
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    
    if args.model == 'dae':
        model.predict(mal_test_dataset_producer, predict_model, indicator_masking=False)
    else:
        model.predict(mal_test_dataset_producer, indicator_masking=False)

    ben_feature_vectors = []
    with torch.no_grad():
        c = args.n_center if args.n_center < ben_count else ben_count
        for ben_x, ben_y in ben_test_dataset_producer:
            ben_x, ben_y = utils.to_tensor(ben_x, ben_y.long(), device=dv)
            ben_feature_vectors.append(ben_x)
            if len(ben_feature_vectors) * hp_params['batch_size'] >= c:
                break
        ben_feature_vectors = torch.vstack(ben_feature_vectors)[:c]

    attack = GDKDE(ben_feature_vectors,
                   args.bandwidth,
                   penalty_factor=args.penalty_factor,
                   kappa=args.kappa,
                   device=model.device
                   )

    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    
    x_adv_samples = []
    
    
    model.eval()
    if args.model == 'dae':
        # 对筛选后的数据进行处理
        for x, y in mal_test_dataset_producer:
            # 数据格式转换和设备迁移
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            
            adv_x_batch = attack.perturb(predict_model, x, y,
                                        args.steps,
                                        args.step_length,
                                        min_lambda_=1e-5,
                                        max_lambda_=1e5,
                                        verbose=True)

            # 对抗样本的数据类型转换
            adv_x_batch = adv_x_batch.to(torch.float32)

            # 使用当前模型清洗对抗样本
            Purified_adv_x_batch = model(adv_x_batch).to(torch.float64)
            
            Purified_adv_x_batch = Purified_adv_x_batch.to(model.device)
            
            # 使用预测模型对清洗后的对抗样本进行预测
            y_cent_batch, _ = predict_model.inference_batch_wise(Purified_adv_x_batch)
            
            y_cent_list.append(y_cent_batch)

        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
        logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / len(y_pred) * 100:.3f}%')
    
    else:       
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            adv_x_batch = attack.perturb(model, x, y,
                                        args.steps,
                                        args.step_length,
                                        min_lambda_=1e-5,
                                        max_lambda_=1e5,
                                        verbose=True)
            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)
            y_cent_list.append(y_cent_batch)
            x_density_list.append(x_density_batch)
            x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
            
            x_adv_samples.append((adv_x_batch).detach().cpu().numpy())
            
        
        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
        logger.info(
            f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    if args.real:
        adv_app_dir = os.path.join(save_dir, 'adv_apps')
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               save_dir=adv_app_dir)
        adv_feature_paths = dataset.apk_preprocess(adv_app_dir, update_feature_extraction=True)

        ben_test_dataset_producer = dataset.get_input_producer(adv_feature_paths,
                                                               np.ones((len(adv_feature_paths,))),
                                                               batch_size=hp_params['batch_size'],
                                                               name='test'
                                                               )
        model.predict(ben_test_dataset_producer, indicator_masking=False)


if __name__ == '__main__':
    _main()
