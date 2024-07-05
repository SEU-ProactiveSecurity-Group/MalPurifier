import os
import multiprocessing
import torch
from torch.nn.modules.module import Module
import numpy as np
from core.droidfeature import InverseDroidFeature
from tools import utils
from config import logging, ErrorHandler

import torch.nn.functional as F

logger = logging.getLogger('examples.base_attack')
logger.addHandler(ErrorHandler)


class BaseAttack(Module):
    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        """
        Initialize the BaseAttack class.

        Parameters:
        - is_attacker (bool): Whether the attack is performed by an attacker (default: True).
        - oblivion (bool): Whether to enable oblivion (default: False).
        - kappa (float): The value of kappa for the attack (default: 1.0).
        - manipulation_x (torch.Tensor or np.ndarray): The manipulation input (default: None).
        - omega (torch.Tensor or np.ndarray): The interdependent APIs (default: None).
        - device (torch.device): The device to use for computation (default: None).
        """
        super(BaseAttack, self).__init__()
        self.is_attacker = is_attacker
        self.oblivion = oblivion
        self.kappa = kappa
        self.manipulation_x = manipulation_x
        self.device = device
        self.omega = omega
        self.inverse_feature = InverseDroidFeature()
        self.initialize()

    def initialize(self):
        # Initialize manipulation_x if not provided
        if self.manipulation_x is None:
            self.manipulation_x = self.inverse_feature.get_manipulation()
        self.manipulation_x = torch.LongTensor(
            self.manipulation_x).to(self.device)

        # Initialize omega if not provided
        if self.omega is None:
            self.omega = self.inverse_feature.get_interdependent_apis()
        self.omega = torch.sum(F.one_hot(torch.tensor(self.omega), num_classes=len(
            self.inverse_feature.vocab)), dim=0).to(self.device)

        # Initialize api_flag
        api_flag = self.inverse_feature.get_api_flag()
        self.api_flag = torch.BoolTensor(api_flag).to(self.device)

    def perturb(self, model, x, adj=None, label=None):
        raise NotImplementedError

    def produce_adv_mal(self, x_mod_list, feature_path_list, app_dir, save_dir=None):
        assert len(x_mod_list) == len(feature_path_list)
        assert isinstance(x_mod_list[0], (torch.Tensor, np.ndarray))

        # Set default save_dir if not provided
        if save_dir is None:
            save_dir = os.path.join('/tmp/', 'adv_mal_cache')

        # Create save_dir if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Convert x_mod_list to instructions
        x_mod_instructions = [self.inverse_feature.inverse_map_manipulation(
            x_mod) for x_mod in x_mod_list]

        # Get app_path_list based on app_dir
        if os.path.isdir(app_dir):
            app_path_list = [os.path.join(app_dir, os.path.basename(
                os.path.splitext(feat_p)[0])) for feat_p in feature_path_list]
        elif isinstance(app_dir, list):
            app_path_list = app_dir
        else:
            raise ValueError(
                "Expected app_dir to be a directory or a list of paths, but got {}.".format(type(app_dir)))

        # Check if all application paths exist
        assert np.all([os.path.exists(app_path)
                      for app_path in app_path_list]), "Cannot find all application paths."

        # Prepare arguments for multiprocessing
        pargs = [(x_mod_instr, feature_path, app_path, save_dir) for x_mod_instr, feature_path, app_path in zip(x_mod_instructions,
                                                                                                                feature_path_list, app_path_list) if not os.path.exists(os.path.join(save_dir, os.path.splitext(os.path.basename(app_path))[0] + '_adv'))]

        # Set the number of CPU cores to use
        cpu_count = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() - \
            2 > 1 else 1

        # Use multiprocessing to produce adversarial malware
        with multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer) as pool:
            for res in pool.map(InverseDroidFeature.modify_wrapper, pargs):
                if isinstance(res, Exception):
                    logger.exception(res)

    def check_lambda(self, model):
        # Check if model has detector enabled and oblivion is False
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            return True
        else:
            return False

    def get_loss(self, model, adv_x, label, lambda_=None):
        # Forward pass through the model
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(adv_x)
        else:
            logits_f = model.forward(adv_x)

        # Calculate cross-entropy loss
        ce = F.cross_entropy(logits_f, label, reduction='none')

        # Get predicted labels
        y_pred = logits_f.argmax(1)

        # Calculate loss based on detector and oblivion settings
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            assert lambda_ is not None
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction = ce + lambda_ * \
                    (torch.clamp(tau - prob_g, max=self.kappa))
            else:
                loss_no_reduction = ce + lambda_ * (tau - prob_g)
            done = (y_pred != label) & (prob_g <= tau)
        else:
            loss_no_reduction = ce
            done = y_pred != label

        return loss_no_reduction, done

    def get_scores(self, model, pertb_x, label, lmda=1.):
        # Forward pass through the model
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(pertb_x)
        else:
            logits_f = model.forward(pertb_x)

        # Get predicted labels
        y_pred = logits_f.argmax(1)

        # Calculate scores based on detector and oblivion settings
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)
            done = (y_pred != label) & (prob_g <= tau)
        else:
            done = y_pred != label

        return done
