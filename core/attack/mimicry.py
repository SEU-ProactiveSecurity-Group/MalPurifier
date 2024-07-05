import torch
import numpy as np
from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.mimicry')
logger.addHandler(ErrorHandler)


class MimicryAttack(BaseAttack):
    def __init__(self, ben_x, oblivion=False, device=None):
        super(MimicryAttack, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, seed=0, is_apk=False, verbose=False):
        assert trials > 0
        if x is None or len(x) <= 0:
            return []

        if len(self.ben_x) <= 0:
            return x

        trials = min(trials, len(self.ben_x))
        success_flag = np.array([])

        with torch.no_grad():
            x_mod_list = []

            for _x in x:
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                _x_fixed_one = (
                    (1. - self.manipulation_x).float() * _x)[None, :]

                modified_x = torch.clamp(
                    _x_fixed_one + trial_vectors, min=0., max=1.)
                modified_x, y = utils.to_tensor(
                    modified_x.double(), torch.ones(trials,).long(), model.device)
                y_cent, x_density = model.inference_batch_wise(modified_x)
                y_pred = np.argmax(y_cent, axis=-1)

                if hasattr(model, 'indicator') and (not self.oblivion):
                    attack_flag = (y_pred == 0) & (
                        model.indicator(x_density, y_pred))
                else:
                    attack_flag = (y_pred == 0)

                ben_id_sel = np.argmax(attack_flag)

                if 'indicator' in type(model).__dict__.keys():
                    use_flag = (y_pred == 0) & (
                        model.indicator(x_density, y_pred))
                else:
                    use_flag = attack_flag

                if not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                else:
                    success_flag = np.append(success_flag, [True])

                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)

            if is_apk:
                return success_flag, np.vstack(x_mod_list)
            else:
                return success_flag, None

    def perturb_dae(self, dae_model, predict_model, x, trials=10, seed=0, is_apk=False, verbose=False):
        assert trials > 0
        if x is None or len(x) <= 0:
            return []

        if len(self.ben_x) <= 0:
            return x

        trials = min(trials, len(self.ben_x))
        success_flag = np.array([])

        with torch.no_grad():
            x_mod_list = []

            for _x in x:
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                _x_fixed_one = (
                    (1. - self.manipulation_x).float() * _x)[None, :]

                modified_x = torch.clamp(
                    _x_fixed_one + trial_vectors, min=0., max=1.)
                modified_x, y = utils.to_tensor(
                    modified_x.double(), torch.ones(trials,).long(), predict_model.device)

                adversarial = modified_x.to(torch.float32)

                outputs = dae_model(adversarial)

                Purified_modified_x = outputs.to(torch.float64)

                modified_x = Purified_modified_x.to(dae_model.device)

                y_cent, x_density = predict_model.inference_batch_wise(
                    modified_x)
                y_pred = np.argmax(y_cent, axis=-1)

                if hasattr(predict_model, 'indicator') and (not self.oblivion):
                    attack_flag = (y_pred == 0) & (
                        predict_model.indicator(x_density, y_pred))
                else:
                    attack_flag = (y_pred == 0)

                ben_id_sel = np.argmax(attack_flag)

                if 'indicator' in type(predict_model).__dict__.keys():
                    use_flag = (y_pred == 0) & (
                        predict_model.indicator(x_density, y_pred))
                else:
                    use_flag = attack_flag

                if not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                else:
                    success_flag = np.append(success_flag, [True])

                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)

            if is_apk:
                return success_flag, np.vstack(x_mod_list)
            else:
                return success_flag, None
