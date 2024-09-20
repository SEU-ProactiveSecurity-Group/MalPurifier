from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np
import torch
import joblib
import pickle

from core.defense import Dataset

from core.defense import MalwareDetectionDNN, PGDAdvTraining, RFGSMAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus, DAE, VAE_SU, \
    MalwareDetectionCNN, MalwareDetectionDT, MalwareDetectionFCN, MalwareDetectionLSTM, MalwareDetectionRF, MalwareDetectionRNN, MalwareDetectionSVM
    
from torch.utils.data import DataLoader, TensorDataset
    
from core.attack import BCA
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.obfuscate_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for bca')

atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='Whether to know the opponent indicator.')

atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='Attack confidence.')

atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='Number of examples loaded in each batch.')

atta_argparse.add_argument('--Ob_type', type=str, default='COMBINE',
                           choices=['API_REFLECTION', 'BENIGN_CLASS', 'COMBINE', 'PCM',
                                    'RESOURCE', 'RM_PERMISSION', 'STRING', 'VARIABLE'],
                           help="Obfuscation type: 'API_REFLECTION', 'BENIGN_CLASS', 'COMBINE', 'PCM',\
                                    'RESOURCE', 'RM_PERMISSION', 'STRING', 'VARIABLE' ")

atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus',
                                    'amd_pad_ma', 'fd_vae', 'dae',
                                    'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'md_at_fgsm', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae', 'dae', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'.")

atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')

atta_argparse.add_argument('--basic_model', type=str, default='md_dnn',
                           choices=['md_dnn', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'],
                           help="'md_dnn', 'md_svm', 'rf', 'md_cnn', 'md_fcn', 'md_lstm', 'dt', 'md_rnn'")

atta_argparse.add_argument('--basic_model_name', type=str, default='20230724-230516',
                           help='dnn basic_dnn_name')


def _main():
    args = atta_argparse.parse_args()
    
    # Select the corresponding model based on the input model parameters and set the save directory
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
    elif args.model == 'rf':
        save_dir = config.get('experiments', 'rf') + '_' + args.model_name
    elif args.model == 'dt':
        save_dir = config.get('experiments', 'dt') + '_' + args.model_name
    elif args.model == 'md_cnn':
        save_dir = config.get('experiments', 'md_cnn') + '_' + args.model_name         
    elif args.model == 'md_svm':
        save_dir = config.get('experiments', 'md_svm') + '_' + args.model_name    
    elif args.model == 'md_rnn':
        save_dir = config.get('experiments', 'md_rnn') + '_' + args.model_name   
    elif args.model == 'md_lstm':
        save_dir = config.get('experiments', 'md_lstm') + '_' + args.model_name  
    elif args.model == 'md_fcn':
        save_dir = config.get('experiments', 'md_fcn') + '_' + args.model_name                                         
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma', 'fd_vae' and 'dae'.")

    # Read hyperparameter configuration
    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))

    # Create dataset
    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
    test_x, testy = dataset.test_dataset

    # Get malicious samples and save, or load if already exists
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)

    # load pkl files
    obfuscated_apk_file = config.get('DEFAULT', 'project_root') + "/obfuscated/" + config.get('DEFAULT', 'dataset_name') + "/x_" + args.Ob_type + ".pkl"
    
    with open(obfuscated_apk_file, "rb") as fr:
        mal_test_x = joblib.load(fr)

        print("Total number of attack samples", len(mal_test_x))
        mal_test_x = np.array(mal_test_x)

    test_z_shape = mal_test_x.shape[0]    # Get the number of samples in test_z

    # Create label sets with all elements set to 1
    mal_test_y = np.ones(test_z_shape, dtype=int)
    
    print("mal_test_x.shape", mal_test_x.shape)
    print("mal_testy.shape", mal_test_y.shape)

    def create_dataloader(mal_test_x, mal_test_y, batch_size):
        # Convert numpy arrays to PyTorch tensors
        mal_test_x_tensor = torch.tensor(mal_test_x, dtype=torch.float32)
        mal_testy_tensor = torch.tensor(mal_test_y, dtype=torch.int64)
        
        # Create TensorDataset
        dataset = TensorDataset(mal_test_x_tensor, mal_testy_tensor)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader

    mal_test_dataset_producer = create_dataloader(mal_test_x, mal_test_y, args.batch_size)

    # Print the total number of malicious samples
    logger.info(f"Total number of malicious samples: {len(mal_test_x)}")
    
    # Check the number of malicious samples
    mal_count = len(mal_test_y)
    if mal_count <= 0:
        return

    # Check the number of classes
    assert dataset.n_classes == 2

    # Check if CUDA is used, if not use CPU
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    
    from pprint import pprint   
    pprint(hp_params)
    pprint(args)
        
    # Create and load model
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    # Create corresponding model based on model type
    # If the model is ICNN or PAD-MA, add advanced processing
    # Adjust model parameters and load model according to different model types
    # Other types directly load the model
    
    if args.model == 'amd_icnn' or args.model == 'amd_pad_ma':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       name=args.model_name,
                                       **hp_params
                                       )
    # Move the model to the specified device (GPU or CPU) and convert to double type
    model = model.to(dv).double()

    # Execute different loading strategies for different model types
    if args.model == 'md_at_pgd':
        at_wrapper = PGDAdvTraining(model)
        at_wrapper.load()  # Load PGD adversarial training model
        model = at_wrapper.model
        
    elif args.model == 'md_at_ma':
        at_wrapper = MaxAdvTraining(model)
        at_wrapper.load()  # Load Max adversarial training model
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
        model.load()  # Load KernelDensityEstimation model
        
    elif args.model == 'amd_dla':
        model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=args.model_name,
                                     **hp_params
                                     )
        model = model.to(dv).double()
        model.load()  # Load AMalwareDetectionDLA model
        
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv).double()
        model.load()  # Load AMalwareDetectionDNNPlus model
        
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()  # Load AMalwareDetectionPAD model
        model = adv_model.model

    elif args.model == 'fd_vae':
        model = VAE_SU(name=args.model_name)
        model.load()
        model.to(dv)
        
    elif args.model == 'rf':
        model = MalwareDetectionRF(name=args.model_name)
        model.load()

    elif args.model == 'dt':
        model = MalwareDetectionDT(name=args.model_name)
        model.load()
        
    elif args.model == 'md_lstm':
        model = MalwareDetectionLSTM(input_dim = dataset.vocab_size,
                                     device = dv,
                                     name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)          
        
    elif args.model == 'md_cnn':
        model = MalwareDetectionCNN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)        
        
    elif args.model == 'md_fcn':
        model = MalwareDetectionFCN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)     
        
    elif args.model == 'md_svm':
        model = MalwareDetectionSVM(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)         
    
    elif args.model == 'md_rnn':
        model = MalwareDetectionRNN(input_size = dataset.vocab_size,
                                    device = dv,
                                    name=args.model_name)
        model = model.to(dv).double()
        model.to(dv)                       

    elif args.model == 'dae':
        model = DAE(input_size = dataset.vocab_size,
            device = dv,
            name = args.model_name,
            predict_model_name = args.basic_model_name
            )
        model.load()
        model.to(dv)
        
        if args.basic_model == 'md_dnn':
            predict_model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=model.predict_model_name,
                                **hp_params
                                )
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_dnn') + '_' + model.predict_model_name,
                                            'model.pth')
        
        elif args.basic_model == 'rf':
            predict_model = MalwareDetectionRF(name=model.predict_model_name)
            predict_model.load()
            predict_model_save_path = os.path.join(config.get('experiments', 'rf') + '_' + model.predict_model_name,
                                            'model.pkl')
        
        elif args.basic_model == 'dt':
            predict_model = MalwareDetectionDT(name=model.predict_model_name)
            predict_model.load()
            predict_model_save_path = os.path.join(config.get('experiments', 'dt') + '_' + model.predict_model_name,
                                            'model.pkl')

        elif args.basic_model == 'md_svm':
            predict_model = MalwareDetectionSVM(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_svm') + '_' + model.predict_model_name,
                                            'model.pth')          
            
        elif args.basic_model == 'md_lstm':  
            predict_model = MalwareDetectionLSTM(input_dim = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_lstm') + '_' + model.predict_model_name,
                                            'model.pth')    
            
        elif args.basic_model == 'md_cnn':  
            predict_model = MalwareDetectionCNN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_cnn') + '_' + model.predict_model_name,
                                            'model.pth')                
            
        elif args.basic_model == 'md_fcn':
            predict_model = MalwareDetectionFCN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_fcn') + '_' + model.predict_model_name,
                                            'model.pth')      
            
        elif args.basic_model == 'md_rnn':
            predict_model = MalwareDetectionRNN(input_size = dataset.vocab_size,
                                        device = dv,
                                        name=model.predict_model_name)
            predict_model = predict_model.to(dv).double()
            predict_model_save_path = os.path.join(config.get('experiments', 'md_rnn') + '_' + model.predict_model_name,
                                            'model.pth')          
            
        print("[⭐] Basic Model: ", predict_model_save_path)
        
        if args.basic_model == 'rf' or args.basic_model == 'dt':
            predict_model.load()
        else:    
            predict_model.load_state_dict(torch.load(predict_model_save_path, map_location=dv))
        predict_model.eval()
        
    else:
        model.load()  # Directly load the model for other types of models

    # Use saved model parameters
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    
    # Make predictions on the test set
    if args.model == 'dae':
        model.predict(mal_test_dataset_producer, predict_model, indicator_masking=False)
    else:
        model.predict(mal_test_dataset_producer, indicator_masking=False)

    # Initialize several empty lists for subsequent data collection
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    
    x_adv_samples = []

    # Set the model to evaluation mode
    model.eval()

    if args.model == 'dae':
        for x, y in mal_test_dataset_producer:
            x, y = utils.to_tensor(x.double(), y.long(), model.device)
            
            adv_x_batch = x.detach().clone().to(torch.double) 

            # Convert adversarial sample data type
            adv_x_batch = adv_x_batch.to(torch.float32)

            # Use the current model to clean the adversarial samples
            Purified_adv_x_batch = model(adv_x_batch).to(torch.float64)

            Purified_adv_x_batch = Purified_adv_x_batch.to(model.device)
            
            # Use the prediction model to predict the cleaned adversarial samples
            y_cent_batch, _ = predict_model.inference_batch_wise(Purified_adv_x_batch)
            y_cent_list.append(y_cent_batch)

        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
        logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / len(y_pred) * 100:.3f}%')
    
    else:
        # Process each sample in the test set
        for x, y in mal_test_dataset_producer:
            # Convert data to tensors
            x, y = utils.to_tensor(x.double(), y.long(), model.device)

            # Perform adversarial attack on the model and get adversarial samples
            adv_x_batch = x.detach().clone().to(torch.double) 

            # Perform inference on the perturbed data
            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)

            # Collect data
            y_cent_list.append(y_cent_batch)
            x_density_list.append(x_density_batch)
            x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
            x_adv_samples.append((adv_x_batch).detach().cpu().numpy())

        # Get prediction results
        y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)

        # Output the average accuracy of adversarial samples
        # The accuracy here refers to the proportion of these samples that the model successfully predicts as malware.
        logger.info(
            f'The mean accuracy on adversarial malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    # If the model has an indicator, calculate the effect of the indicator
    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        # Calculate and output the effectiveness of the indicator
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        # Calculate and output the model's prediction accuracy for adversarial malware samples, considering the results of the indicator.
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

if __name__ == '__main__':
    _main()