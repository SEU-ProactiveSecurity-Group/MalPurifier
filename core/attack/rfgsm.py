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
from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x, or_tensors
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.rfgsm')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class RFGSM(BaseAttack):
    def __init__(self, is_attacker=True, random=False, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(RFGSM, self).__init__(is_attacker, oblivion,
                                    kappa, manipulation_x, omega, device)
        self.omega = None
        self.manipulation_z = None
        self.lmba = 1.
        self.random = random

    def _perturb(self, model, x, label=None, steps=10, step_length=0.02, lmbda=1., use_sample=False):
        if x is None or x.shape[0] <= 0:
            return []

        adv_x = x.clone()
        model.eval()
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)
        loss_natural = 0.

        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmbda)

            if t == 0:
                loss_natural = loss

            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)
            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)

            adv_x = torch.clamp(adv_x + step_length *
                                torch.sign(grad4ins_), min=0., max=1.)

        round_threshold = torch.rand(adv_x.size()).to(
            self.device) if self.random else 0.5
        adv_x = round_x(adv_x, round_threshold)
        adv_x = or_tensors(adv_x, x)

        loss_adv, _1 = self.get_loss(model, adv_x, label, lmbda)

        replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(adv_x)
        adv_x[replace_flag] = x[replace_flag]

        return adv_x

    def perturb(self, model, x, label=None, steps=10, step_length=0.02, min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False, use_sample=False):
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lmba = min_lambda_
        else:
            self.lmba = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lmba <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lmba)

            if torch.all(done):
                break

            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, step_length, lmbda=self.lmba, use_sample=use_sample)
            adv_x[~done] = pert_x
            self.lmba *= base

        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lmba)

        if verbose:
            logger.info(
                f"rFGSM: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x

    def perturb_dae(self, model, purifier, x, label=None, steps=10, step_length=0.02, min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False, use_sample=False, oblivion=False):
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lmba = min_lambda_
        else:
            self.lmba = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lmba <= max_lambda_:
            with torch.no_grad():
                if not oblivion:
                    purified_adv = purifier(
                        adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_loss(
                        model, purified_adv, label, self.lmba)
                else:
                    _, done = self.get_loss(model, adv_x, label, self.lmba)

            if torch.all(done):
                break

            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, step_length, lmda=self.lmba, use_sample=use_sample)
            adv_x[~done] = pert_x
            self.lmba *= base

        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lmba)

        if verbose:
            logger.info(
                f"rFGSM: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x
