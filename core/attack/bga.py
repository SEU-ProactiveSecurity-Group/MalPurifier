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
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, xor_tensors, or_tensors
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.bga')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BGA(BaseAttack):
    """
    Multi-step bit gradient ascent

    Parameters
    ---------
    @param is_attacker, Boolean, if true means the role is the attacker
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence on adversary indicator
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BGA, self).__init__(is_attacker, oblivion,
                                  kappa, manipulation_x, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are permitted to be insertable
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None, m=10, lmda=1., use_sample=False):
        """
        Perturb the feature vector of the node.

        Parameters:
        ----------
        model : PyTorch model
            The target model being attacked.
        x : torch.FloatTensor
            The feature vector with shape [batch_size, vocab_dim].
        label : torch.LongTensor, optional
            The true labels of the data.
        m : int, default 10
            The maximum number of perturbations, corresponding to hp k in the paper.
        lmda : float, default 1.0
            The penalty factor balancing the importance of the adversary detector.
        use_sample : bool, default False
            Whether to use a random starting point.

        Returns:
        ----------
        worst_x : torch.FloatTensor
            The perturbed feature vector.
        """

        # Check if input x is empty or invalid
        if x is None or x.shape[0] <= 0:
            return []

        # Initialize
        sqrt_m = torch.from_numpy(
            np.sqrt([x.size()[1]])).float().to(model.device)
        adv_x = x.clone()
        worst_x = x.detach().clone()

        # Set the model to evaluation mode
        model.eval()

        # Get the starting point for perturbation
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)

        # Perform m perturbations
        for t in range(m):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmda)

            # Save the effective perturbation results
            worst_x[done] = adv_x[done]

            # Compute gradients
            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            # Compute update
            x_update = (sqrt_m * (1. - 2. * adv_x) * grad >=
                        torch.norm(grad, 2, 1).unsqueeze(1).expand_as(adv_x)).float()

            # Perturb the feature vector
            adv_x = xor_tensors(x_update, adv_x)
            adv_x = or_tensors(adv_x, x)

            # Select adversarial samples
            done = self.get_scores(model, adv_x, label).data
            worst_x[done] = adv_x[done]

        return worst_x

    def perturb(self, model, x, label=None, steps=10, min_lambda_=1e-5, max_lambda_=1e5, use_sample=False, base=10., verbose=False):
        """
        Perturb the input data x to change its output on the given model.

        Parameters:
        ----------
        model : PyTorch model
            The target model to be attacked.
        x : torch.Tensor
            The input data.
        label : torch.Tensor, optional
            The true labels of the input data.
        steps : int, default 10
            The number of attack steps.
        min_lambda_ : float, default 1e-5
            The minimum value of attack strength.
        max_lambda_ : float, default 1e5
            The maximum value of attack strength.
        use_sample : bool, default False
            Whether to use samples.
        base : float, default 10.0
            The base used to adjust lambda.
        verbose : bool, default False
            Whether to print detailed information.

        Returns:
        ----------
        adv_x : torch.Tensor
            The perturbed input data.
        """

        # Ensure the range of lambda is valid
        assert 0 < min_lambda_ <= max_lambda_

        # Set the model to evaluation mode
        model.eval()

        # Set the initial value of lambda based on the model attribute
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # Create a perturbable version with the same shape as x
        adv_x = x.detach().clone().to(torch.double)

        # Perturb within the specified lambda range
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if torch.all(done):
                break
            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, lmda=self.lambda_, use_sample=use_sample)
            adv_x[~done] = pert_x
            self.lambda_ *= base

        # Get the final loss value
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)

            # Print attack effectiveness if verbose is set
            if verbose:
                logger.info(
                    f"BGA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")

        return adv_x
