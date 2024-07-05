"""
@ARTICLE{9321695,
  author={D. {Li} and Q. {Li} and Y. F. {Ye} and S. {Xu}},
  journal={IEEE Transactions on Network Science and Engineering},
  title={A Framework for Enhancing Deep Neural Networks against Adversarial Malware},
  year={2021},
  doi={10.1109/TNSE.2021.3051354}}
"""

import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.pgdl1')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class PGDl1(BaseAttack):
    """
    Projected gradient descent (ascent) with gradients 'normalized' using l1 norm.
    By comparing BCA, the api removal is leveraged

    Parameters
    ---------
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(PGDl1, self).__init__(is_attacker, oblivion,
                                    kappa, manipulation_x, omega, device)
        self.lambda_ = 1.

    def _perturb(self, model, x,  label=None,
                 steps=10,
                 lambda_=1.):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        worst_x = x.detach().clone()
        self.lambda_ = lambda_
        # we set a graph contains two apis at least
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1
        model.eval()
        if "rnn" in model.model_save_path:
            model.train()
        if "lstm" in model.model_save_path:
            model.train()
        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, self.lambda_)
            worst_x[done] = adv_x[done]
            if torch.all(done):
                break
            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            if grads[0] is None:
                # Handle the situation where the gradient is None.
                # For example, you can set the gradient to zeros:
                grad = torch.zeros_like(var_adv_x)
            else:
                grad = grads[0].data

            perturbation, direction = self.get_perturbation(grad, x, adv_x)
            # stop perturbing the examples that are successful to evade the victim
            perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation *
                                direction, min=0., max=1.)
        done = self.get_scores(model, adv_x, label)
        worst_x[done] = adv_x[done]
        return worst_x

    def perturb(self, model, x, label=None,
                steps=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_
        adv_x = x.detach().clone().to(torch.double)
        while self.lambda_ <= max_lambda_:
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if torch.all(done):
                break
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                   steps,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(
                    f"pgd l1: attack effectiveness {done.sum().item() / x.size()[0]}.")
        return adv_x

    def perturb_dae(self, model, purifier, x, label=None,
                    steps=10,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    base=10.,
                    verbose=False,
                    oblivion=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_
        adv_x = x.detach().clone().to(torch.double)
        while self.lambda_ <= max_lambda_:
            if not oblivion:
                purified_adv = purifier(
                    adv_x.detach().clone().float()).to(torch.double)
                _, done = self.get_loss(
                    model, purified_adv, label, self.lambda_)
            else:
                _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if torch.all(done):
                break
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                   steps,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lambda_)
            if verbose:
                logger.info(
                    f"pgd l1: attack effectiveness {done.sum().item() / x.size()[0]}.")
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        # 1. mask paddings
        gradients = gradients * self.padding_mask

        # 2. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    2.1 api insertion
        pos_insertion = (adv_features <= 0.5) * 1
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        #    2.2 api removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients <= 0) * (pos_removal &
                                           self.manipulation_x) * gradients
        if self.is_attacker:
            #     2.2.1 cope with the interdependent apis
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(
                gradients * checking_nonexist_api, dim=-1, keepdim=True)

        gradients = grad4removal + grad4insertion

        # 3. remove duplications
        un_mod = torch.abs(features - adv_features) <= 1e-6
        gradients = gradients * un_mod

        # 4. look for important position
        absolute_grad = torch.abs(gradients).reshape(features.shape[0], -1)
        _, position = torch.max(absolute_grad, dim=-1)
        perturbations = F.one_hot(
            position, num_classes=absolute_grad.shape[-1]).double()
        perturbations = perturbations.reshape(features.shape)
        directions = torch.sign(gradients) * (perturbations > 1e-6)

        # 5. tailor the interdependent apis
        if self.is_attacker:
            perturbations += (torch.any(directions[:, self.api_flag]
                              < 0, dim=-1, keepdim=True)) * checking_nonexist_api
            directions += perturbations * self.omega
        return perturbations, directions
