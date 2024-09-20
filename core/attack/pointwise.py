from tqdm import tqdm
import torch
import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.Pointwise')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class Pointwise(BaseAttack):
    """
    Pointwise attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param ben_x: torch.FloatTensor, feature vectors with shape [number_of_benign_files, vocab_dim]
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_x, oblivion=False, device=None):
        super(Pointwise, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, repetition=10, max_eta=1, epsilon=1000, seed=0, is_apk=False, verbose=False):
        assert repetition > 0

        if x is None or len(x) <= 0:
            return []

        if len(self.ben_x) <= 0:
            return x

        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flags = np.array([])
        x_mod_list = []
        x_adv_init = []

        # For the pointwise attack, we first continuously enhance the generation of salt-and-pepper noises
        # and add them to the malicious sample, increasing the noise intensity by 1/1000 each time
        # until the classifier misclassifies it as benign or Repeat the process 10 times to generate
        # adversarial samples by modifying the features.
        # Pointwise attack strategy

        def salt_and_pepper_noise(x, model, max_eta=1, repetition=10, epsilon=1000):
            device = model.device
            torch.manual_seed(seed)

            perturbed_x = x.clone()
            for _ in range(repetition):
                use_flag = False
                for eta in torch.linspace(0, max_eta, epsilon):
                    uni_noises = torch.rand(perturbed_x.shape).to(device)
                    salt = (uni_noises >= 1. - eta / 2).float()
                    pepper = -(uni_noises < eta / 2).float()

                    perturbed_x += salt + pepper
                    perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)

                perturbed_x, y = utils.to_tensor(
                    perturbed_x.double(), torch.ones(trials,).long(), device)

                y_cent, x_density = model.inference_batch_wise(perturbed_x)
                y_pred = np.argmax(y_cent, axis=-1)

                if hasattr(model, 'indicator') and not self.oblivion:
                    use_flag = (y_pred == 0) & model.indicator(
                        x_density, y_pred)
                else:
                    use_flag = (y_pred == 0)

                if self.oblivion:
                    continue
                elif use_flag:
                    break

            return perturbed_x

        with torch.no_grad():
            x_adv_init = [salt_and_pepper_noise(
                xi, model, max_eta, repetition, epsilon) for xi in x]
            x_adv_init = torch.stack(x_adv_init).detach().cpu().numpy()

        adv_x_list = []
        with torch.no_grad():
            torch.manual_seed(seed)
            x_adv_init = torch.tensor(np.array(x_adv_init))
            x_adv = x_adv_init.clone()

            success_flag = []

            x_len = len(x)
            for idx, (original, adversarial) in enumerate(tqdm(zip(x, x_adv_init), total=x_len, desc="Attacking")):
                original = original.flatten()
                adversarial = adversarial.flatten()

                found = False
                indices = torch.randperm(len(original))

                for i in indices:
                    tmp_value = adversarial[i]
                    adversarial[i] = original[i]

                    adversarial_tensor = adversarial.clone().detach().to(
                        model.device).to(torch.float64)

                    y_cent, x_density = model.inference_batch_wise(
                        adversarial_tensor)
                    prediction = np.argmax(y_cent, axis=-1)

                    if hasattr(model, 'indicator') and not self.oblivion:
                        use_flag = (prediction == 0) & model.indicator(
                            x_density, prediction)
                    else:
                        use_flag = (prediction == 0)

                    if use_flag:
                        found = True
                        break
                    else:
                        adversarial[i] = tmp_value

                success_flag.append(found)
                x_adv[idx] = adversarial.reshape(x_adv[idx].shape)

            x_adv_tensor = torch.stack(
                [item.clone().detach().to(x.device) for item in x_adv])
            x_mod = (x_adv_tensor - x).cpu().numpy()
            x_mod_list.append(x_mod)
            adv_x_list.append(x_adv_tensor.detach().cpu().numpy())

        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None

    def perturb_dae(self, dae_model, predict_model, x, trials=10, repetition=10, max_eta=1, epsilon=1000, seed=0, is_apk=False, verbose=False):
        assert trials > 0

        if x is None or len(x) <= 0:
            return []

        if len(self.ben_x) <= 0:
            return x

        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flags = np.array([])
        x_mod_list = []
        x_adv_init = []

        def salt_and_pepper_noise(x, model, max_eta=1, repetition=10, epsilon=1000):
            device = model.device
            torch.manual_seed(seed)

            perturbed_x = x.clone()
            for _ in range(repetition):
                use_flag = False
                for eta in torch.linspace(0, max_eta, epsilon):
                    uni_noises = torch.rand(perturbed_x.shape).to(device)
                    salt = (uni_noises >= 1. - eta / 2).float()
                    pepper = -(uni_noises < eta / 2).float()

                    perturbed_x += salt + pepper
                    perturbed_x = torch.clamp(perturbed_x, min=0., max=1.)

                perturbed_x, y = utils.to_tensor(
                    perturbed_x.double(), torch.ones(trials,).long(), device)

                y_cent, x_density = predict_model.inference_batch_wise(
                    perturbed_x)
                y_pred = np.argmax(y_cent, axis=-1)

                if hasattr(model, 'indicator') and not self.oblivion:
                    use_flag = (y_pred == 0) & model.indicator(
                        x_density, y_pred)
                else:
                    use_flag = (y_pred == 0)

                if self.oblivion:
                    continue
                elif use_flag:
                    break

            return perturbed_x

        success_flags = np.array(success_flags)

        with torch.no_grad():
            x_adv_init = [salt_and_pepper_noise(
                xi, dae_model, max_eta, repetition, epsilon) for xi in x]
            x_adv_init = torch.stack(x_adv_init).detach().cpu().numpy()

        adv_x_list = []
        with torch.no_grad():
            torch.manual_seed(seed)
            x_adv_init = torch.tensor(np.array(x_adv_init))
            x_adv = x_adv_init.clone()

            success_flag = []

            x_len = len(x)
            for idx, (original, adversarial) in enumerate(zip(x, x_adv_init)):
                original = original.flatten()
                adversarial = adversarial.flatten()

                found = False
                indices = torch.randperm(len(original))

                for i in indices:
                    tmp_value = adversarial[i]
                    adversarial[i] = original[i]

                    adversarial_tensor = adversarial.clone().detach().to(
                        dae_model.device).to(torch.float32).unsqueeze(0)

                    Purified_adv_x_batch = dae_model(adversarial_tensor)

                    outputs_numpy = Purified_adv_x_batch.cpu().numpy()
                    reshape_encoded_data = np.where(outputs_numpy >= 0.5, 1, 0)

                    Purified_modified_x = torch.tensor(
                        reshape_encoded_data, device=dae_model.device, dtype=torch.float64).squeeze(0)

                    y_cent, x_density = predict_model.inference_batch_wise(
                        Purified_modified_x)
                    prediction = np.argmax(y_cent, axis=-1)

                    if prediction == 0:
                        found = True
                        break
                    else:
                        adversarial[i] = tmp_value

                success_flag.append(found)
                x_adv[idx] = adversarial.reshape(x_adv[idx].shape)

            x_adv_tensor = torch.stack(
                [item.clone().detach().to(x.device) for item in x_adv])
            x_mod = (x_adv_tensor - x).cpu().numpy()
            x_mod_list.append(x_mod)
            adv_x_list.append(x_adv_tensor.detach().cpu().numpy())

        if is_apk:
            return success_flags, np.vstack(adv_x_list), np.vstack(x_mod_list)
        else:
            return success_flags, np.vstack(adv_x_list), None
