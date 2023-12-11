"""
@InProceedings{10.1007/978-3-642-40994-3_25,
author="Biggio, Battista
and Corona, Igino
and Maiorca, Davide
and Nelson, Blaine
and {\v{S}}rndi{\'{c}}, Nedim
and Laskov, Pavel
and Giacinto, Giorgio
and Roli, Fabio",
editor="Blockeel, Hendrik
and Kersting, Kristian
and Nijssen, Siegfried
and {\v{Z}}elezn{\'y}, Filip",
title="Evasion Attacks against Machine Learning at Test Time",
booktitle="Machine Learning and Knowledge Discovery in Databases",
year="2013"
}
"""

import torch
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.gdkdel1')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class GDKDEl1(BaseAttack):
    """
    a variant of gradient descent with kernel density estimation: we calculate the density estimation upon the hidden
    space and perturb the feature in the direction of l1 norm based gradients

    Parameters
    ---------
    @param benign_feat: torch.Tensor, representation of benign files on the feature space
    @param bandwidth: float, variance of gaussian distribution
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, float, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, benign_feat=None, bandwidth=20., penalty_factor=1000.,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(GDKDEl1, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.benign_feat = benign_feat
        self.bandwidth = bandwidth
        self.penalty_factor = penalty_factor
        self.lambda_ = 1.
        if isinstance(self.benign_feat, torch.Tensor):
            pass
        elif isinstance(self.benign_feat, np.ndarray):
            self.benign_feat = torch.tensor(self.benign_feat, device=device)
        else:
            raise TypeError

    def _perturb(self, model, x, label=None,
                 steps=50,
                 lambda_=1.):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.DoubleTensor, feature vectors with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param m: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        model.eval()
        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label)
            if torch.all(done):
                break
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0]
            perturbation, direction = self.get_perturbation(grad, x, adv_x)
            # avoid to perturb the examples that are successful to evade the victim
            perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation * direction, min=0., max=1.)
        return adv_x

    def perturb(self, model, x, label=None,
                steps=50,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()

        if hasattr(model, 'forward_g'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_
        adv_x = x.detach().clone().to(torch.double)
        while self.lambda_ <= max_lambda_:
            _, done = self.get_loss(model, adv_x, label)
            if torch.all(done):
                break
            adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                   steps,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label)
            if verbose:
                logger.info(f"gdkdel1: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3}%.")
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):

        # look for allowable position, because only '1--> -' and '0 --> +' are permitted
        # api insertion
        pos_insertion = (adv_features <= 0.5) * 1
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        # api removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients <= 0) * (pos_removal & self.manipulation_x) * gradients
        # cope with the interdependent apis
        if self.is_attacker:
            # cope with the interdependent apis
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True)

        gradients = grad4removal + grad4insertion

        # remove duplications (i.e., neglect the positions whose values have been modified previously.)
        un_mod = torch.abs(features - adv_features) <= 1e-6
        gradients = gradients * un_mod

        # look for important position
        absolute_grad = torch.abs(gradients).reshape(features.shape[0], -1)
        _, position = torch.max(absolute_grad, dim=-1)
        perturbations = F.one_hot(position, num_classes=absolute_grad.shape[-1]).double()
        perturbations = perturbations.reshape(features.shape)
        directions = torch.sign(gradients) * (perturbations > 1e-6)

        if self.is_attacker:
            perturbations += (torch.any(directions[:, self.api_flag] < 0, dim=-1, keepdim=True)) * checking_nonexist_api
            directions += perturbations * self.omega
        return perturbations, directions

    def get_loss(self, model, adv_x, label):
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(adv_x)
        else:
            logits_f = model.forward(adv_x)
        ce = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        # square = torch.sum(torch.square(self.benign_feat.float().unsqueeze(dim=0) -
        #                                 adv_x.float().unsqueeze(dim=1)),
        #                    dim=-1)
        square = torch.stack([torch.sum(torch.square(ben_x - adv_x.float()), dim=-1) for ben_x in self.benign_feat.float()],
                             dim=-1)
        kde = torch.mean(torch.exp(-square / (self.bandwidth ** 2)), dim=-1)
        loss_no_reduction = ce - self.penalty_factor * kde
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction += self.lambda_ * (torch.clamp(tau - prob_g, max=self.kappa))
            else:
                loss_no_reduction += self.lambda_ * (tau - prob_g)
            done = (y_pred != label) & (prob_g <= tau)
        else:
            done = y_pred != label
        return loss_no_reduction, done
