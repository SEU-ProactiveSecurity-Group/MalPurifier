"""
@inproceedings{grosse2017adversarial,
  title={Adversarial examples for malware detection},
  author={Grosse, Kathrin and Papernot, Nicolas and Manoharan, Praveen and Backes, Michael and McDaniel, Patrick},
  booktitle={European symposium on research in computer security},
  pages={62--79},
  year={2017},
  organization={Springer}
}
"""

import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.grosse')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class Groose(BaseAttack):
    """
    Multi-step bit coordinate ascent applied upon softmax output (rather than upon the loss)

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
        super(Groose, self).__init__(is_attacker, oblivion,
                                     kappa, manipulation_x, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are insertable
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None, steps=10, lambda_=1.):
        """
        Perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.DoubleTensor, feature vectors with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of perturbations
        @param lambda_: float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []

        adv_x = x
        worst_x = x.detach().clone()
        model.eval()

        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, _done = self.get_loss(model, var_adv_x, 1 - label)
            worst_x[_done] = adv_x[_done]

            if torch.all(_done):
                break

            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)
            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)

            _2, pos = torch.max(grad4ins_, dim=-1)
            perturbation = F.one_hot(
                pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)

            # Stop perturbing the examples that are successful to evade the victim
            perturbation[_done] = 0.
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)

        # Select adv x
        done = self.get_scores(model, adv_x, label).data
        worst_x[done] = adv_x[done]
        return worst_x

    def perturb(self, model, x, label=None, steps=50, min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False):
        """
        Enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lambda_ <= max_lambda_:
            _, done = self.get_loss(model, adv_x, 1 - label)

            if torch.all(done):
                break

            # Recompute the perturbation under other penalty factors
            adv_x[~done] = x[~done]
            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, lambda_=self.lambda_)
            adv_x[~done] = pert_x
            self.lambda_ *= base

        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, 1 - label)

            if verbose:
                logger.info(
                    f"grosse: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

        return adv_x

    def perturb_dae(self, model, purifier, x, label=None, steps=50, min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False, oblivion=False):
        """
        Enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        model.eval()
        purifier.eval()

        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        adv_x = x.detach().clone().to(torch.double)

        while self.lambda_ <= max_lambda_:
            if not oblivion:
                purified_adv = purifier(
                    adv_x.detach().clone().float()).to(torch.double)
                _, done = self.get_loss(model, purified_adv, 1 - label)
            else:
                _, done = self.get_loss(model, adv_x, 1 - label)

            if torch.all(done):
                break

            # Recompute the perturbation under other penalty factors
            adv_x[~done] = x[~done]
            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, lambda_=self.lambda_)
            adv_x[~done] = pert_x
            self.lambda_ *= base

        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, 1 - label)

            if verbose:
                logger.info(
                    f"grosse: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

        return adv_x

    def get_loss(self, model, adv_x, tar_label):
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(adv_x)
        else:
            logits_f = model.forward(adv_x)

        softmax_loss = torch.softmax(
            logits_f, dim=-1)[torch.arange(tar_label.size()[0]), tar_label]
        y_pred = logits_f.argmax(1)

        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)

            if self.is_attacker:
                loss_no_reduction = softmax_loss + self.lambda_ * \
                    (torch.clamp(tau - prob_g, max=self.kappa))
            else:
                loss_no_reduction = softmax_loss + \
                    self.lambda_ * (tau - prob_g)

            done = (y_pred == tar_label) & (prob_g <= tau)
        else:
            loss_no_reduction = softmax_loss
            done = y_pred == tar_label

        return loss_no_reduction, done
