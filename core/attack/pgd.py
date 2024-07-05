"""
@ARTICLE{9321695,
  author={D. {Li} and Q. {Li} and Y. F. {Ye} and S. {Xu}},
  journal={IEEE Transactions on Network Science and Engineering},
  title={A Framework for Enhancing Deep Neural Networks against Adversarial Malware},
  year={2021},
  doi={10.1109/TNSE.2021.3051354}}
"""

import torch

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import config, logging, ErrorHandler
from tools import utils
import os


logger = logging.getLogger('core.attack.pgd')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class PGD(BaseAttack):
    """
    Projected gradient descent (ascent).

    Parameters
    ---------
    @param norm, 'l2' or 'linf'
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, norm, use_random=False, rounding_threshold=0.5,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(PGD, self).__init__(is_attacker, oblivion,
                                  kappa, manipulation_x, omega, device)
        assert norm == 'l1' or norm == 'l2' or norm == 'linf', "Expect 'l1', 'l2' or 'linf'."
        self.norm = norm
        self.use_random = use_random
        assert 0 < rounding_threshold < 1
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None, steps=10, step_length=1., lambda_=1.):
        """
        Perturb node feature vectors based on given parameters.

        Parameters:
        -----------
        model : torch.nn.Module
            The victim model that will be used for perturbation.

        x : torch.FloatTensor
            Node feature vectors (each represents the occurrences of APIs in a graph) 
            with shape [batch_size, vocab_dim].

        label : torch.LongTensor, optional
            Ground truth labels of the input data.

        steps : int, optional
            Maximum number of iterations for perturbation.

        step_length : float, optional
            The step size in each iteration for perturbation.

        lambda_ : float, optional
            Penalty factor for the loss calculation.

        Returns:
        --------
        adv_x : torch.FloatTensor
            Perturbed node feature vectors.
        """

        # Return an empty list if input data is invalid
        if x is None or x.shape[0] <= 0:
            return []

        # Initialize adversarial example as the original input
        adv_x = x
        self.lambda_ = lambda_

        # Set the model to evaluation mode
        model.eval()
        if "rnn" in model.model_save_path:
            model.train()
        if "lstm" in model.model_save_path:
            model.train()
        loss_natural = 0.
        for t in range(steps):
            # If using randomness, initialize the adversarial example in the first iteration
            if t == 0 and self.use_random:
                adv_x = get_x0(
                    adv_x, rounding_threshold=self.round_threshold, is_sample=True)

            # Convert adv_x to a Variable for gradient calculation
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)

            # Compute the loss of the adversarial example
            loss, done = self.get_loss(model, var_adv_x, label, self.lambda_)
            if t == 0:
                loss_natural = loss

            # Calculate the gradient of the loss with respect to the adversarial example
            grads = torch.autograd.grad(
                loss.mean(), var_adv_x, allow_unused=True)
            grad = grads[0].data

            # Compute perturbations based on the gradient
            perturbation = self.get_perturbation(grad, x, adv_x)

            # Apply perturbations to the adversarial example
            adv_x = torch.clamp(adv_x + perturbation *
                                step_length, min=0., max=1.)

        # Rounding of the adversarial example after all iterations
        if self.norm == 'linf' and (not hasattr(model, 'is_detector_enabled')):
            round_threshold = torch.rand(x.size()).to(self.device)
        else:
            round_threshold = self.round_threshold
        adv_x = round_x(adv_x, round_threshold)

        # Compute loss for the final adversarial example
        loss_adv, _ = self.get_loss(model, adv_x, label, self.lambda_)

        # Replace adversarial examples that don't increase loss
        replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(adv_x)
        adv_x[replace_flag] = x[replace_flag]

        return adv_x

    def perturb(self, model, x, label=None, steps=10, step_length=1., min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False):
        """
        Enhance the attack by perturbing node feature vectors based on given parameters.

        Parameters:
        -----------
        model : torch.nn.Module
            The victim model that will be used for perturbation.

        x : torch.FloatTensor
            Node feature vectors to be perturbed.

        label : torch.LongTensor, optional
            Ground truth labels of the input data.

        steps : int, optional
            Maximum number of iterations for perturbation.

        step_length : float, optional
            The step size in each iteration for perturbation.

        min_lambda_ : float, optional
            Minimum penalty factor for loss calculation.

        max_lambda_ : float, optional
            Maximum penalty factor for loss calculation.

        base : float, optional
            Scaling factor for lambda_.

        verbose : bool, optional
            Flag to determine if logging should be verbose.

        Returns:
        --------
        adv_x : torch.FloatTensor
            Perturbed node feature vectors.
        """
        # Check if the given lambdas are valid
        assert 0 < min_lambda_ <= max_lambda_

        # Check if the model has an attribute 'k' which triggers dense graph issue
        if 'k' in list(model.__dict__.keys()) and model.k > 0:
            logger.warning(
                "The attack leads to dense graph and triggers the issue of out of memory.")

        # Basic validation for steps and step length
        assert steps >= 0 and step_length >= 0

        # Set the model to evaluation mode
        model.eval()

        # Set the initial lambda based on the model's attributes
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # Clone the input data to avoid modifying it in-place
        adv_x = x.detach().clone().to(torch.double)

        # Loop to enhance the attack as long as lambda is within its allowed range
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lambda_)

            # Break if all the examples are successfully perturbed
            if torch.all(done):
                break

            # Perturb the feature vectors that aren't yet successfully attacked
            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, step_length, lambda_=self.lambda_)
            adv_x[~done] = pert_x

            # Increase lambda for the next iteration
            self.lambda_ *= base

        # Compute the final attack effectiveness
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                effectiveness = done.sum().item() / done.size()[0] * 100
                logger.info(
                    f"pgd {self.norm}: attack effectiveness {effectiveness:.3f}%.")

        return adv_x

    def perturb_dae(self, model, purifier, x, label=None, steps=10, step_length=1., min_lambda_=1e-5, max_lambda_=1e5, base=10., verbose=False, oblivion=False):

        # Check if the given lambdas are valid
        assert 0 < min_lambda_ <= max_lambda_

        # Check if the model has an attribute 'k' which triggers dense graph issue
        if 'k' in list(model.__dict__.keys()) and model.k > 0:
            logger.warning(
                "The attack leads to dense graph and triggers the issue of out of memory.")

        # Basic validation for steps and step length
        assert steps >= 0 and step_length >= 0

        # Set the model to evaluation mode
        model.eval()
        purifier.eval()

        # Set the initial lambda based on the model's attributes
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        # Clone the input data to avoid modifying it in-place
        adv_x = x.detach().clone().to(torch.double)

        # Loop to enhance the attack as long as lambda is within its allowed range
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                if not oblivion:
                    purified_adv = purifier(
                        adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_loss(
                        model, purified_adv, label, self.lambda_)
                else:
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)

            # Break if all the examples are successfully perturbed
            if torch.all(done):
                break

            # Perturb the feature vectors that aren't yet successfully attacked
            pert_x = self._perturb(
                model, adv_x[~done], label[~done], steps, step_length, lambda_=self.lambda_)
            adv_x[~done] = pert_x

            # Increase lambda for the next iteration
            self.lambda_ *= base

        # Compute the final attack effectiveness
        with torch.no_grad():
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(model, purified_adv, label, self.lambda_)
            if verbose:
                effectiveness = done.sum().item() / done.size()[0] * 100
                logger.info(
                    f"pgd {self.norm}: attack effectiveness {effectiveness:.3f}%.")

        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        """
        Calculate perturbations based on gradients and features.

        Args:
        - gradients: Gradient tensor
        - features: Original features tensor
        - adv_features: Adversarial features tensor

        Returns:
        - perturbation: Perturbation tensor
        """

        # Define allowable positions for API insertion and removal based on constraints.
        # For API insertion: only '1--> -' and '0 --> +' are permitted
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)
        grad4insertion = (gradients >= 0) * pos_insertion * gradients

        # For API removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients < 0) * (pos_removal &
                                          self.manipulation_x) * gradients

        # Handle interdependent APIs for attackers
        if self.is_attacker:
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(
                gradients * checking_nonexist_api, dim=-1, keepdim=True)

        gradients = grad4removal + grad4insertion

        # Compute perturbations based on chosen norm
        if self.norm == 'linf':
            perturbation = torch.sign(gradients)
        elif self.norm == 'l2':
            l2norm = torch.linalg.norm(gradients, dim=-1, keepdim=True)
            perturbation = torch.minimum(
                torch.tensor(1., dtype=features.dtype, device=features.device),
                gradients / l2norm
            )

            # Handle cases with NaN or Inf values
            perturbation = torch.where(
                torch.isnan(perturbation), 0., perturbation)
            perturbation = torch.where(
                torch.isinf(perturbation), 1., perturbation)
        else:
            raise ValueError("Expect 'l2' or 'linf' norm.")

        # Add extra perturbations due to interdependent APIs
        if self.is_attacker:
            if self.norm == 'linf':
                perturbation += torch.any(
                    perturbation[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api
            elif self.norm == 'l2':
                min_val = torch.amin(perturbation, dim=-1,
                                     keepdim=True).clamp_(max=0.)
                perturbation += (torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                                 keepdim=True) * torch.abs(min_val) * checking_nonexist_api)

        return perturbation
