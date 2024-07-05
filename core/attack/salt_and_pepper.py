import torch

import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.salt_and_pepper')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class Salt_and_pepper(BaseAttack):
    def __init__(self, ben_x, oblivion=False, device=None):
        super(Salt_and_pepper, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, epsilon=10, max_eta=0.001, repetition=10, seed=0, is_apk=False, verbose=False):
        if x is None or len(x) <= 0:
            return []

        if len(self.ben_x) <= 0:
            return x

        trials = min(trials, len(self.ben_x))

        device = model.device
        torch.manual_seed(seed)

        success_flags = []
        x_mod_list = []
        adv_x_list = []

        with torch.no_grad():
            for _x in x:
                shape = _x.shape
                perturbed_x = _x.clone()

                for _ in range(repetition):
                    for eta in torch.linspace(0, max_eta, steps=min(epsilon, shape[0]))[1:]:
                        uni_noises = torch.rand(perturbed_x.shape).to(device)
                        salt = (uni_noises >= 1. - eta / 2).float()
                        pepper = -(uni_noises < eta / 2).float()

                        perturbed_x += salt + pepper
                        perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)
                        perturbed_x = utils.round_x(perturbed_x, 0.5)

                        if hasattr(model, 'indicator') and not self.oblivion:
                            y_cent, x_density = model.inference_batch_wise(
                                perturbed_x)
                            use_flag = (y_pred == 0) & model.indicator(
                                x_density, y_pred)
                            if use_flag:
                                break

                    y_cent, x_density = model.inference_batch_wise(perturbed_x)
                    y_pred = np.argmax(y_cent, axis=-1)

                    if hasattr(model, 'indicator') and not self.oblivion:
                        use_flag = (y_pred == 0) & model.indicator(
                            x_density, y_pred)
                    else:
                        use_flag = (y_pred == 0)

                    if use_flag:
                        break

                success_flags.append(use_flag)
                x_mod = (perturbed_x - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)
                adv_x_list.append(perturbed_x.detach().cpu().numpy())

        success_flags = np.array(success_flags)

        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None
