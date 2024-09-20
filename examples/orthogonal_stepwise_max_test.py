from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, PGDAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus
from core.attack import OrthogonalStepwiseMax
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.orthogonal_stepwise_max_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for step-wise max attack')
atta_argparse.add_argument('--steps', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_check', type=int, default=1,
                           help='check the maximum at $step_check$th step..')
atta_argparse.add_argument('--project_detector', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--project_classifier', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--step_length_l1', type=float, default=1.,
                           help='step length in each step of pgd l1.')
atta_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                           help='step length in each step of pgd l2.')
atta_argparse.add_argument('--step_length_linf', type=float, default=0.01,
                           help='step length in each step of pgd linf.')
atta_argparse.add_argument('--random_start', action='store_true', default=False,
                           help='randomly initialize the start points.')
atta_argparse.add_argument('--round_threshold', type=float, default=0.5,
                           help='threshold for rounding real scalars at the initialization step.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='number of examples loaded in per batch.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus', 'amd_pad_ma'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx',
                           help='model timestamp.')


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
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', and 'amd_pad_ma'.")

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
    if mal_count <= 0:
        return
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=args.batch_size,
                                                           name='test')
    assert dataset.n_classes == 2

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    # initial model
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
    else:
        model.load()
        
    logger.info("Load model parameters from {}.".format(model.model_save_path))

    # Predict the given malicious sample set, where indicator_masking=False indicates no indicator masking
    model.predict(mal_test_dataset_producer, indicator_masking=False)

    # Initialize OrthogonalStepwiseMax attack method
    attack = OrthogonalStepwiseMax(project_detector=args.project_detector,
                                project_classifier=args.project_classifier,
                                k=None, use_random=args.random_start,
                                rounding_threshold=args.round_threshold,
                                device=model.device
                                )

    # Initialize lists to store subsequent intermediate results
    y_cent_list, x_density_list = [], []  # Store model inference results
    x_mod_integrated = []  # Store the difference between each perturbed sample and the original sample
    x_adv_samples = []  # Store perturbed malicious samples

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the malicious test dataset and perform adversarial attacks on each sample
    for x, y in mal_test_dataset_producer:
        # Convert data and labels to tensors and move to the appropriate device
        x, y = utils.to_tensor(x.double(), y.long(), model.device)

        # Perturb samples using given attack parameters
        adv_x_batch = attack.perturb(model, x, y,
                                    args.steps,
                                    args.step_check,
                                    args.step_length_l1,
                                    args.step_length_l2,
                                    args.step_length_linf,
                                    verbose=True)

        # Perform inference on perturbed samples to get center predictions and density values
        y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)

        # Save center predictions, density values, differences, and samples themselves to their respective lists
        y_cent_list.append(y_cent_batch)
        x_density_list.append(x_density_batch)
        x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
        x_adv_samples.append((adv_x_batch).detach().cpu().numpy())

    # Evaluate prediction accuracy of perturbed samples
    y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
    logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    # If the model includes an indicator method, evaluate and print the effectiveness of the indicator
    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    # Define the directory to save adversarial samples and ensure it exists
    save_dir = os.path.join(config.get('experiments', 'orthogonal_stepwise_max'), args.model)
    if not os.path.exists(save_dir):
        utils.mkdir(save_dir)

    # Convert difference list and adversarial sample list to numpy arrays
    x_mod_integrated = np.concatenate(x_mod_integrated, axis=0)
    x_adv_samples = np.concatenate(x_adv_samples, axis=0)
    print("⭐ x_adv_samples.shape:", x_adv_samples.shape)

    # Define labels for adversarial samples (all set to 1)
    test_z_labels = np.ones(x_adv_samples.shape[0], dtype=int)

    # Save adversarial samples and labels as a pkl file
    import pickle
    with open(os.path.join(save_dir, "x_adv.pkl"), "wb") as fw:
        pickle.dump((x_adv_samples, test_z_labels), fw)

    # Save sample differences in perturbation space
    utils.dump_pickle_frd_space(x_mod_integrated, os.path.join(save_dir, 'x_mod.list'))

    # If set to real, generate real adversarial malware using perturbed samples
    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(), config.get('dataset', 'malware_dir'))


if __name__ == '__main__':
    _main()
