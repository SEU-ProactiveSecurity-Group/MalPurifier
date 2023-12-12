from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np
import torch

from core.defense import Dataset  # 导入防御模块中的Dataset类

from core.defense import MalwareDetectionDNN, PGDAdvTraining, RFGSMAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE, VAE_SU
    
from core.attack import Mimicry
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.mimicry')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for mimicry attack')
atta_argparse.add_argument('--trials', type=int, default=5,
                           help='number of benign samples for perturbing one malicious file.')
atta_argparse.add_argument('--n_ben', type=int, default=5000,
                           help='number of benign samples.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus',
                                    'amd_pad_ma', 'fd_vae', 'dae'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae', 'dae'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')
atta_argparse.add_argument('--basic_dnn_name', type=str, default='20231016-121617',
                           help='basic_dnn_name')

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
    val_dataset_producer = dataset.get_input_producer(*dataset.validation_dataset, batch_size=hp_params['batch_size'],
                                                      name='val')
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)
                
    # 打印出总共恶意样本的数量
    logger.info(f"⭐Total number of malicious samples: {len(mal_test_x)}")    
    
    
        
    mal_count = len(mal_testy)
    ben_test_x, ben_testy = test_x[testy == 0], testy[testy == 0]
    ben_count = len(ben_test_x)
    if mal_count <= 0 and ben_count <= 0:
        return
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=hp_params['batch_size'],
                                                           name='test')
    ben_test_dataset_producer = dataset.get_input_producer(ben_test_x, ben_testy,
                                                           batch_size=hp_params['batch_size'],
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
        predict_model = predict_model.to(dv).double()
        predict_model_save_path = os.path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        print("Basic DNN Model: ", predict_model_save_path)
        predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()        
        
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    model.eval()
    
    if args.model == 'dae':
        model.predict(mal_test_dataset_producer, predict_model, indicator_masking=False)
    else:
        model.predict(mal_test_dataset_producer, indicator_masking=False)
        
    ben_feature_vectors = []
    with torch.no_grad():
        c = args.n_ben if args.n_ben < ben_count else ben_count
        for ben_x, ben_y in ben_test_dataset_producer:
            ben_x, ben_y = utils.to_tensor(ben_x.double(), ben_y.long(), device=dv)
            ben_feature_vectors.append(ben_x)
            if len(ben_feature_vectors) * hp_params['batch_size'] >= c:
                break
        ben_feature_vectors = torch.vstack(ben_feature_vectors)[:c]

    attack = Mimicry(ben_feature_vectors, oblivion=args.oblivion, device=model.device)
    success_flag_list = []
    x_mod_list = []
    
    if args.model == 'dae':
        # 对每个恶意样本执行模仿攻击
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            # 注意，此处调用的函数为：perturb_dae
            _flag, x_mod = attack.perturb_dae(model,
                                        predict_model,
                                        x,
                                        trials=args.trials,
                                        seed=0,
                                        is_apk=args.real,
                                        verbose=True)
            success_flag_list.append(_flag)
            logger.info(
                f"The attack effectiveness under mimicry attack is {np.sum(_flag) / float(len(_flag)) * 100}%.")
            x_mod_list.append(x_mod)
        # 统计整体的攻击成功率
        success_flag = np.concatenate(success_flag_list)
        logger.info(f"The mean accuracy on perturbed malware is {(1. - np.sum(success_flag) / float(mal_count)) * 100}%.")    
    else:
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            _flag, x_mod = attack.perturb(model,
                                        x,
                                        trials=args.trials,
                                        seed=0,
                                        is_apk=args.real,
                                        verbose=True)
            success_flag_list.append(_flag)
            logger.info(
                f"The attack effectiveness under mimicry attack is {np.sum(_flag) / float(len(_flag)) * 100}%.")
            x_mod_list.append(x_mod)
        success_flag = np.concatenate(success_flag_list)
        logger.info(f"The mean accuracy on perturbed malware is {(1. - np.sum(success_flag) / float(mal_count)) * 100}%.")

    # save_dir = os.path.join(config.get('experiments', 'mimicry'), args.model)
    # if not os.path.exists(save_dir):
    #     utils.mkdir(save_dir)

    if args.real:
        x_mod_list = np.concatenate(x_mod_list, axis=0)
        utils.dump_pickle_frd_space(x_mod_list,
                                    os.path.join(save_dir, 'x_mod.list'))

        adv_app_dir = os.path.join(save_dir, 'adv_apps')
        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)

        # x_mod_list = utils.read_pickle_frd_space(os.path.join(save_dir, 'x_mod.list'))
        attack.produce_adv_mal(x_mod_list, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               save_dir=adv_app_dir)

        adv_feature_paths = dataset.apk_preprocess(adv_app_dir, update_feature_extraction=True)
        # dataset.feature_preprocess(adv_feature_paths)
        adv_test_dataset_producer = dataset.get_input_producer(adv_feature_paths,
                                                               np.ones((len(adv_feature_paths, ))),
                                                               batch_size=hp_params['batch_size'],
                                                               name='test'
                                                               )
        model.predict(adv_test_dataset_producer, indicator_masking=False)



if __name__ == '__main__':
    _main()
