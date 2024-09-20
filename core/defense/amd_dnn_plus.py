"""
@article{grosse2017statistical,
  title={On the (statistical) detection of adversarial examples},
  author={Grosse, Kathrin and Manoharan, Praveen and Papernot, Nicolas and Backes, Michael and McDaniel, Patrick},
  journal={arXiv preprint arXiv:1702.06280},
  year={2017}
}

@inproceedings{carlini2017adversarial,
  title={Adversarial examples are not easily detected: Bypassing ten detection methods},
  author={Carlini, Nicholas and Wagner, David},
  booktitle={Proceedings of the 10th ACM workshop on artificial intelligence and security},
  pages={3--14},
  year={2017}
}

This implementation is adapted from:
https://github.com/carlini/nn_breaking_detection
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from core.attack.max import Max
from core.attack.stepwise_max import StepwiseMax
from core.defense.md_dnn import MalwareDetectionDNN
from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_dnn_plus')
logger.addHandler(ErrorHandler)


class AMalwareDetectionDNNPlus(nn.Module, DetectorTemplate):
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.95,
                 device='cpu', name='', **kwargs):
        nn.Module.__init__(self)
        DetectorTemplate.__init__(self)

        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = ratio
        self.device = device
        self.name = name
        self.parse_args(**kwargs)

        if md_nn_model is not None and isinstance(md_nn_model, nn.Module):
            self.md_nn_model = md_nn_model
            self.is_fitting_md_model = False
        else:
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   self.n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            self.is_fitting_md_model = True

        self.amd_nn_plus = MalwareDetectionDNN(self.input_size,
                                               self.n_classes + 1,
                                               self.device,
                                               name,
                                               **kwargs)

        self.tau = nn.Parameter(torch.zeros(
            [1, ], device=self.device), requires_grad=False)

        self.model_save_path = path.join(config.get('experiments', 'amd_dnn_plus') + '_' + self.name,
                                         'model.pth')
        self.md_nn_model.model_save_path = self.model_save_path

        logger.info('========================================NN_PLUS model architecture==============================')
        logger.info(self)
        logger.info('===============================================end==============================================')

    def parse_args(self,
                   dense_hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
                   **kwargs
                   ):
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
        else:
            raise TypeError("Expect a list of hidden units.")

        self.dropout = dropout
        self.alpha_ = alpha_
        self.proc_number = kwargs['proc_number']

        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        logits = self.amd_nn_plus(x)
        logits -= torch.amax(logits, dim=-1, keepdim=True).detach()

        if logits.dim() == 1:
            return logits, torch.softmax(logits, dim=-1)[-1]
        else:
            return logits, torch.softmax(logits, dim=-1)[:, -1]

    def predict(self, test_data_producer, indicator_masking=True):
        y_cent, x_prob, y_true = self.inference(test_data_producer)

        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        indicator_flag = self.indicator(x_prob).cpu().numpy()

        def measurement(_y_true, _y_pred):
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score

            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info(f"The accuracy on the test dataset is {accuracy * 100:.5f}%")
            logger.info(f"The balanced accuracy on the test dataset is {b_accuracy * 100:.5f}%")

            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("class absent.")
                return

            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(_y_true, _y_pred, average='binary')

            logger.info(f"False Negative Rate (FNR) is {fnr * 100:.5f}%, \
                        False Positive Rate (FPR) is {fpr * 100:.5f}%, F1 score is {f1 * 100:.5f}%")

        measurement(y_true, y_pred)

        if indicator_masking:
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            y_pred[~indicator_flag] = 1.

        logger.info('The indicator is turning on...')
        logger.info(f'The threshold is {self.tau.item():.5}')

        measurement(y_true, y_pred)

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits, x_cent = self.forward(x)
                y_cent.append(F.softmax(logits, dim=-1)[:, :2])
                x_prob.append(x_cent)
                gt_labels.append(y)

        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        return y_cent, x_prob, gt_labels

    def inference_batch_wise(self, x):
        assert isinstance(x, torch.Tensor)
        self.eval()
        logits, g = self.forward(x)
        softmax_output = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        if len(softmax_output.shape) == 1:
            x_cent = softmax_output[:2]
        else:
            x_cent = softmax_output[:, :2]

        return x_cent, g.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau

    def indicator(self, x_prob, y_pred=None):
        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob <= self.tau
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    def get_threshold(self, validation_data_producer, ratio=None):
        self.eval()

        ratio = ratio if ratio is not None else self.ratio

        assert 0 <= ratio <= 1

        probabilities = []
        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.device)

                _1, x_cent = self.forward(x_val)

                probabilities.append(x_cent)

            s, _ = torch.sort(torch.cat(probabilities, dim=0))

            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0

            self.tau[0] = s[i]

    def fit(self, train_data_producer, validation_data_producer, attack, attack_param,
            epochs=50, lr=0.005, weight_decay=0., verbose=True):
        if self.is_fitting_md_model:
            self.md_nn_model.fit(train_data_producer,
                                 validation_data_producer, epochs, lr, weight_decay)

        if attack is not None:
            assert isinstance(attack, (Max, StepwiseMax))
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker

        optimizer = optim.Adam(self.amd_nn_plus.parameters(),
                               lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        pertb_train_data_list = []
        pertb_val_data_list = []
        nbatches = len(train_data_producer)
        logger.info("Training model with extra class ...")
        self.md_nn_model.eval()

        for i in range(epochs):
            self.amd_nn_plus.train()
            losses, accuracies = [], []

            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train, y_train = utils.to_device(x_train.double(), y_train.long(), self.device)

                start_time = time.time()
                if idx_batch >= len(pertb_train_data_list):
                    pertb_x = attack.perturb(self.md_nn_model, x_train, y_train,
                                             **attack_param
                                             )
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)
                    trivial_atta_flag = torch.sum(torch.abs(x_train - pertb_x), dim=-1)[:] == 0.
                    if torch.all(trivial_atta_flag):
                        pertb_train_data_list.append([])
                        continue
                    pertb_x = pertb_x[~trivial_atta_flag]
                    pertb_train_data_list.append(pertb_x.detach().cpu().numpy())
                else:
                    pertb_x = pertb_train_data_list[idx_batch]
                    if len(pertb_x) == 0:
                        continue
                    pertb_x = torch.from_numpy(pertb_x).to(self.device)

                x_train = torch.cat([x_train, pertb_x], dim=0)
                batch_size_ext = x_train.shape[0]
                y_train = torch.cat([y_train, 2 * torch.ones((pertb_x.shape[0],), dtype=torch.long, device=self.device)])
                idx = torch.randperm(batch_size_ext)
                x_train = x_train[idx]
                y_train = y_train[idx]

                optimizer.zero_grad()
                logits, _1 = self.forward(x_train)
                loss_train = F.cross_entropy(logits, y_train)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time
                acc_train = (logits.argmax(1) == y_train).sum().item()
                acc_train = acc_train / (len(x_train))
                losses.append(loss_train.item())
                accuracies.append(acc_train)

                if verbose:
                    mins, secs = int(total_time / 60), int(total_time % 60)
                    logger.info(f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}%.')

            self.amd_nn_plus.eval()

            avg_acc_val = []

            for idx, (x_val, y_val) in enumerate(validation_data_producer):
                x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)

                if idx >= len(pertb_val_data_list):
                    pertb_x = attack.perturb(self.md_nn_model, x_val, y_val,
                                             **attack_param
                                             )
                    pertb_x = utils.round_x(pertb_x, alpha=0.5)
                    trivial_atta_flag = torch.sum(torch.abs(x_val - pertb_x), dim=-1)[:] == 0.
                    assert (not torch.all(trivial_atta_flag)), 'No modifications.'
                    pertb_x = pertb_x[~trivial_atta_flag]
                    pertb_val_data_list.append(pertb_x.detach().cpu().numpy())
                else:
                    pertb_x = torch.from_numpy(pertb_val_data_list[idx]).to(self.device)

                x_val = torch.cat([x_val, pertb_x], dim=0)
                y_val = torch.cat([y_val, 2 * torch.ones((pertb_x.shape[0],), device=self.device)])

                logits, _1 = self.forward(x_val)
                acc_val = (logits.argmax(1) == y_val).sum().item()
                acc_val = acc_val / (len(x_val))
                avg_acc_val.append(acc_val)

            avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                self.get_threshold(validation_data_producer)
                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                logger.info(f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        assert path.exists(self.model_save_path), 'train model first'
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        torch.save(self.state_dict(), self.model_save_path)
