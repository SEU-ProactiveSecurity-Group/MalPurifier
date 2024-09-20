import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.max')
logger.addHandler(ErrorHandler)
EXP_OVER_FLOW = 1e-30


class Max(BaseAttack):
    """
    Max attack: Iteratively selects results from multiple attack methods.

    Parameters
    --------
    @param attack_list: List, list of instantiated attack objects.
    @param varepsilon: Float, scalar used to determine convergence.
    """

    def __init__(self, attack_list, varepsilon=1e-20,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        """
        Constructor

        Parameters:
        - attack_list: List of instantiated attack objects, should have at least one attack method.
        - varepsilon: Scalar used to determine convergence, default is 1e-20.
        - is_attacker: Bool, indicates if it's an attacker, default is True.
        - oblivion: Bool, a boolean flag (function not specified here), default is False.
        - kappa: Float, a float parameter, default is 1.
        - manipulation_x: Possibly related to data processing or manipulation, specific use not detailed.
        - omega: Specific use of parameter omega not detailed.
        - device: Device, e.g., 'cuda' or 'cpu', used for computation.

        Note:
        - During initialization, it first checks if `attack_list` contains at least one attack object.
        """
        super(Max, self).__init__(is_attacker, oblivion, kappa,
                                  manipulation_x, omega, device)
        assert len(attack_list) > 0, 'At least one attack method is required.'
        self.attack_list = attack_list
        self.varepsilon = varepsilon
        self.device = device

    def perturb(self, model, x, label=None, steps_max=5, min_lambda_=1e-5, max_lambda_=1e5, verbose=False):
        """
        Perturb node features

        Parameters
        -----------
        @param model: Victim model.
        @param x: torch.FloatTensor, feature vector of shape [batch_size, vocab_dim].
        @param label: torch.LongTensor, true labels.
        @param steps_max: Integer, maximum number of iterations.
        @param min_lambda_: float, balances the importance of the adversary detector (if present).
        @param max_lambda_: float, same as above.
        @param verbose: Boolean, whether to print detailed logs.

        Returns
        --------
        adv_x: Perturbed data.
        """

        if x is None or x.shape[0] <= 0:
            return []

        model.eval()

        with torch.no_grad():
            loss, done = self.get_scores(model, x, label)

        pre_loss = loss

        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        adv_x = x.detach().clone()

        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)

        for t in range(steps_max):
            num_sample_red = n - torch.sum(stop_flag)

            if num_sample_red <= 0:
                break

            red_label = label[~stop_flag]
            pertbx = []

            for attack in self.attack_list:
                assert 'perturb' in type(attack).__dict__.keys()

                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False

                if 'Orthogonal' in type(attack).__name__:
                    pertbx.append(attack.perturb(
                        model=model, x=adv_x[~stop_flag], label=red_label))
                else:
                    pertbx.append(attack.perturb(model=model, x=adv_x[~stop_flag], label=red_label,
                                                 min_lambda_=1e-5,
                                                 max_lambda_=1e5,
                                                 ))
            pertbx = torch.vstack(pertbx)

            with torch.no_grad():
                red_label_ext = torch.cat([red_label] * len(self.attack_list))

                loss, done = self.get_scores(model, pertbx, red_label_ext)

                loss = loss.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)

                success_flag = torch.any(done, dim=-1)

                done[~torch.any(done, dim=-1)] = 1

                loss = (loss * done.to(torch.float)) + \
                    torch.min(loss) * (~done).to(torch.float)

                pertbx = pertbx.reshape(
                    len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])

                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(
                    num_sample_red), indices]

                a_loss = loss[torch.arange(num_sample_red), indices]

                pre_stop_flag = stop_flag.clone()

                stop_flag[~stop_flag] = (
                    torch.abs(pre_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag

                pre_loss[~pre_stop_flag] = a_loss

            if verbose:
                with torch.no_grad():
                    _, done = self.get_scores(model, adv_x, label)
                    logger.info(
                        f"max: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

            return adv_x

    def perturb_dae(self, predict_model, purifier, x, label=None, steps_max=5, min_lambda_=1e-5, max_lambda_=1e5, verbose=False, oblivion=False):
        """
        Perturb node features

        Parameters
        -----------
        @param model: Victim model.
        @param x: torch.FloatTensor, feature vector of shape [batch_size, vocab_dim].
        @param label: torch.LongTensor, true labels.
        @param steps_max: Integer, maximum number of iterations.
        @param min_lambda_: float, balances the importance of the adversary detector (if present).
        @param max_lambda_: float, same as above.
        @param verbose: Boolean, whether to print detailed logs.

        Returns
        --------
        adv_x: Perturbed data.
        """

        if x is None or x.shape[0] <= 0:
            return []

        predict_model.eval()
        purifier.eval()

        with torch.no_grad():
            if not oblivion:
                purified_x = purifier(
                    x.detach().clone().float()).to(torch.double)
            else:
                purified_x = x.detach().clone()
            loss, done = self.get_scores(predict_model, purified_x, label)

        pre_loss = loss

        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        adv_x = x.detach().clone()

        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)

        for t in range(steps_max):
            num_sample_red = n - torch.sum(stop_flag)

            if num_sample_red <= 0:
                break

            red_label = label[~stop_flag]
            pertbx = []

            for attack in self.attack_list:
                assert 'perturb' in type(attack).__dict__.keys()

                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False

                if 'Orthogonal' in type(attack).__name__:
                    pertbx.append(attack.perturb_dae(predict_model=predict_model, purifier=purifier,
                                  x=adv_x[~stop_flag], label=red_label, oblivion=oblivion))
                else:
                    pertbx.append(attack.perturb_dae(model=predict_model, purifier=purifier, x=adv_x[~stop_flag], label=red_label,
                                                     min_lambda_=1e-5,
                                                     max_lambda_=1e5,
                                                     oblivion=oblivion
                                                     ))

            pertbx = torch.vstack(pertbx)

            with torch.no_grad():
                red_label_ext = torch.cat([red_label] * len(self.attack_list))

                if not oblivion:
                    purified_pertbx = purifier(
                        pertbx.detach().clone().float()).to(torch.double)
                else:
                    purified_pertbx = pertbx.detach().clone()

                loss, done = self.get_scores(
                    predict_model, purified_pertbx, red_label_ext)

                loss = loss.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list),
                                    num_sample_red).permute(1, 0)

                success_flag = torch.any(done, dim=-1)

                done[~torch.any(done, dim=-1)] = 1

                loss = (loss * done.to(torch.float)) + \
                    torch.min(loss) * (~done).to(torch.float)

                pertbx = pertbx.reshape(
                    len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])

                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(
                    num_sample_red), indices]

                a_loss = loss[torch.arange(num_sample_red), indices]

                pre_stop_flag = stop_flag.clone()

                stop_flag[~stop_flag] = (
                    torch.abs(pre_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag

                pre_loss[~pre_stop_flag] = a_loss

            if verbose:
                with torch.no_grad():
                    purified_adv_x = purifier(
                        adv_x.detach().clone().float()).to(torch.double)
                    _, done = self.get_scores(
                        predict_model, purified_adv_x, label)
                    logger.info(
                        f"max: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")

            return adv_x

    def get_scores(self, model, pertb_x, label):
        """
        Get loss values and prediction completion status for perturbed data on the model.

        Parameters:
        @param model: Model object, the target model being attacked.
        @param pertb_x: torch.Tensor, perturbed data.
        @param label: torch.Tensor, true labels of the perturbed data.

        Returns:
        - loss_no_reduction: Loss value for each sample (without dimension reduction).
        - done: Boolean Tensor, indicates whether the model's prediction for each sample is successfully completed.
        """
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(pertb_x)
        else:
            logits_f = model.forward(pertb_x)

        ce = F.cross_entropy(logits_f, label, reduction='none')

        y_pred = logits_f.argmax(1)

        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)
            loss_no_reduction = -prob_g
            done = (y_pred != label) & (prob_g <= tau)
        else:
            loss_no_reduction = ce
            done = y_pred != label

        return loss_no_reduction, done
