"""
@inproceedings{al2018adversarial,
  title={Adversarial deep learning for robust detection of binary encoded malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O’Reilly, Una-May},
  booktitle={2018 IEEE Security and Privacy Workshops (SPW)},
  pages={76--82},
  year={2018},
  organization={IEEE}
}
"""

import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.bca')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BCA(BaseAttack):
    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BCA, self).__init__(is_attacker, oblivion,
                                  kappa, manipulation_x, omega, device)
        self.omega = None
        self.manipulation_z = None
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None, steps=10, lmda=1., use_sample=False):
        """
        Perturbs the input data to generate adversarial examples.

        Args:
            model (torch.nn.Module): The model to attack.
            x (torch.Tensor): The input data.
            label (torch.Tensor): The true labels of the input data.
            steps (int): The number of perturbation steps.
            lmda (float): The lambda value for controlling the perturbation magnitude.
            use_sample (bool): Whether to use sampling for rounding threshold.

        Returns:
            torch.Tensor: The perturbed input data.
        """
        if x is None or x.shape[0] <= 0:
            return []

        adv_x = x
        worst_x = x.detach().clone()
        model.eval()
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)

        if "rnn" in model.model_save_path:
            model.train()
        if "lstm" in model.model_save_path:
            model.train()

        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmda)

            worst_x[done] = adv_x[done]

            if torch.all(done):
                break

            grads = torch.autograd.grad(loss.mean(), var_adv_x)
            grad = grads[0].data

            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)

            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            _2, pos = torch.max(grad4ins_, dim=-1)

            perturbation = F.one_hot(
                pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)

            perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)

        done = self.get_scores(model, adv_x, label)
        worst_x[done] = adv_x[done]

        return worst_x

    def perturb(self, model, x, label=None, steps=10, min_lambda_=1e-5, max_lambda_=1e5, use_sample=False, base=10., verbose=False):
        """
        Generates adversarial examples using the Basic Cloning Attack (BCA) method.

        Args:
            model (torch.nn.Module): The model to attack.
            x (torch.Tensor): The input data.
            label (torch.Tensor): The true labels of the input data.
            steps (int): The number of perturbation steps.
            min_lambda_ (float): The minimum lambda value for controlling the perturbation magnitude.
            max_lambda_ (float): The maximum lambda value for controlling the perturbation magnitude.
            use_sample (bool): Whether to use sampling for rounding threshold.
            base (float): The base value for increasing lambda.
            verbose (bool): Whether to print attack effectiveness.

        Returns:
            torch.Tensor: The adversarial examples.
        """
        assert 0 < min_lambda_ <= max_lambda_

        model.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lambda_)

            if torch.all(done):
                break

            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, lmda=self.lambda_, use_sample=use_sample)

            adv_x[~done] = pert_x
            self.lambda_ *= base

        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)

            if verbose:
                logger.info(
                    f"BCA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x

    def perturb_dae(self, model, purifier, x, label=None, steps=10, min_lambda_=1e-5, max_lambda_=1e5, use_sample=False, base=10., verbose=False, oblivion=False):
        """
        Generates adversarial examples using the Basic Cloning Attack (BCA) method with a Denoising Autoencoder (DAE) purifier.

        Args:
            model (torch.nn.Module): The model to attack.
            purifier (torch.nn.Module): The DAE purifier.
            x (torch.Tensor): The input data.
            label (torch.Tensor): The true labels of the input data.
            steps (int): The number of perturbation steps.
            min_lambda_ (float): The minimum lambda value for controlling the perturbation magnitude.
            max_lambda_ (float): The maximum lambda value for controlling the perturbation magnitude.
            use_sample (bool): Whether to use sampling for rounding threshold.
            base (float): The base value for increasing lambda.
            verbose (bool): Whether to print attack effectiveness.
            oblivion (bool): Whether to use oblivion mode.

        Returns:
            torch.Tensor: The adversarial examples.
        """
        assert 0 < min_lambda_ <= max_lambda_

        model.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                if not oblivion:
                    purified_adv = purifier(
                        adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_loss(
                        model, purified_adv, label, self.lambda_)
                else:
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)

            if torch.all(done):
                break

            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, lmda=self.lambda_, use_sample=use_sample)

            adv_x[~done] = pert_x
            self.lambda_ *= base

        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lambda_)

            if verbose:
                logger.info(
                    f"BCA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x
