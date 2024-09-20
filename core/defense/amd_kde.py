from os import path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_kde')
logger.addHandler(ErrorHandler)

# Class for anomaly detection using kernel density estimation on the penultimate layer


class KernelDensityEstimation(DetectorTemplate):
    """
    Kernel density estimation on the penultimate layer

    Parameters
    -------------
    @param model: torch.nn.Module, an instance of a model object
    @param bandwidth: float, variance of the Gaussian density function
    @param n_classes: int, number of classes
    @param ratio: float [0,1], ratio for calculating the threshold
    """

    def __init__(self, model, n_centers=1000, bandwidth=20., n_classes=2, ratio=0.9):
        super(KernelDensityEstimation, self).__init__()
        assert isinstance(model, torch.nn.Module)

        self.model = model
        self.device = model.device
        self.n_centers = n_centers
        self.bandwidth = bandwidth
        self.n_classes = n_classes
        self.ratio = ratio
        self.gaussian_means = None

        self.tau = nn.Parameter(torch.zeros(
            [self.n_classes, ], device=self.device), requires_grad=False)
        self.name = self.model.name
        self.model.load()

        self.model_save_path = path.join(config.get(
            'experiments', 'amd_kde').rstrip('/') + '_' + self.name, 'model.pth')
        self.model.model_save_path = self.model_save_path

    def forward(self, x):
        for dense_layer in self.model.dense_layers[:-1]:
            x = self.model.activation_func(dense_layer(x))

        logits = self.model.dense_layers[-1](x)

        if len(logits.shape) == 1:
            x_prob = self.forward_g(x, logits.argmax().detach())
        else:
            x_prob = self.forward_g(x, logits.argmax(1).detach())

        return logits, x_prob

    def forward_f(self, x):
        for dense_layer in self.model.dense_layers[:-1]:
            x = self.model.activation_func(dense_layer(x))
        logits = self.model.dense_layers[-1](x)
        return logits, x

    def forward_g(self, x_hidden, y_pred):
        """
        Calculate the probability of hidden layer representation using kernel density estimation.

        Parameters
        -----------
        @param x_hidden: torch.tensor, hidden representation of nodes
        @param y_pred: torch.tensor, prediction results
        """

        if len(x_hidden.shape) == 1:
            x_hidden = x_hidden.unsqueeze(0)

        size = x_hidden.size()[0]

        dist = [torch.sum(torch.square(means.unsqueeze(dim=0) - x_hidden.unsqueeze(dim=1)), dim=-1) for means in
                self.gaussian_means]

        kd = torch.stack(
            [torch.mean(torch.exp(-d / self.bandwidth ** 2), dim=-1) for d in dist], dim=1)

        return -1 * kd[torch.arange(size), y_pred]

    def get_threshold(self, validation_data_producer, ratio=None):
        """
        Get the threshold for kernel density estimation
        :@param validation_data_producer: object, iterator for generating validation dataset
        :@param ratio: threshold calculation ratio
        """

        ratio = ratio if ratio is not None else self.ratio
        assert 0 <= ratio <= 1

        self.eval()
        probabilities = []
        gt_labels = []

        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(
                    x_val.double(), y_val.long(), self.device)
                logits, x_prob = self.forward(x_val)
                probabilities.append(x_prob)
                gt_labels.append(y_val)

            prob = torch.cat(probabilities, dim=0)
            gt_labels = torch.cat(gt_labels)

            for i in range(self.n_classes):
                prob_x_y = prob[gt_labels == i]
                s, _ = torch.sort(prob_x_y)
                self.tau[i] = s[int((s.shape[0] - 1) * ratio)]

    def predict(self, test_data_producer, indicator_masking=True):
        """
        Make predictions on test data and evaluate results.
        :@param test_data_producer: object, iterator for generating test data
        :@param indicator_masking: boolean, whether to use indicator for filtering
        """

        y_cent, x_prob, y_true = self.inference(test_data_producer)

        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        indicator_flag = self.indicator(x_prob, y_pred).cpu().numpy()
        if indicator_masking:
            flag_of_retaining = indicator_flag
            y_pred = y_pred[flag_of_retaining]
            y_true = y_true[flag_of_retaining]
        else:
            y_pred[~indicator_flag] = 1.

        logger.info('The indicator is turning on...')

        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score

        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)
        logger.info(
            "The accuracy on the test dataset is {:.5f}%".format(accuracy * 100))
        logger.info("The balanced accuracy on the test dataset is {:.5f}%".format(
            b_accuracy * 100))

        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        print("Other evaluation metrics we may need:")
        logger.info("False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
                    .format(fnr * 100, fpr * 100, f1 * 100))

    def eval(self):
        self.model.eval()

    def inference(self, test_data_producer):
        """
        Perform inference on given test data to get class centers, probability values, and true labels.
        :@param test_data_producer: object, iterator for generating test data
        :return: class centers, probability values, and true labels
        """

        y_cent, x_prob = []
        gt_labels = []

        self.eval()

        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_tensor(x.double(), y.long(), self.device)

                logits_f, logits_g = self.forward(x)

                y_cent.append(F.softmax(logits_f, dim=-1))
                x_prob.append(logits_g)
                gt_labels.append(y)

        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)

        return y_cent, x_prob, gt_labels

    def inference_batch_wise(self, x):
        """
        Batch-wise inference function. Performs inference on given input tensor and returns softmax output and probability values.
        :@param x: torch.Tensor, input data tensor
        :return: softmax output and probability values
        """

        assert isinstance(x, torch.Tensor)

        logits, x_prob = self.forward(x)

        return torch.softmax(logits, dim=-1).detach().cpu().numpy(), x_prob.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau[y_pred]

    def indicator(self, x_prob, y_pred=None):
        """
        Indicator function, determines if probability values x_prob are less than or equal to tau values for given sample predictions.
        :@param x_prob: probability values, can be numpy array or torch tensor.
        :@param y_pred: predicted labels, should be non-empty.
        :return: boolean array or tensor, indicating whether each value in x_prob is less than or equal to corresponding tau value.
        """

        assert y_pred is not None

        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob <= self.get_tau_sample_wise(y_pred)).cpu().numpy()

        elif isinstance(x_prob, torch.Tensor):
            return x_prob <= self.get_tau_sample_wise(y_pred)

        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    def fit(self, train_dataset_producer, val_dataset_producer):
        """
        Train and set up the model based on the given training dataset, and determine thresholds for kernel density estimation.

        :@param train_dataset_producer: generator for the training dataset.
        :@param val_dataset_producer: generator for the validation dataset.
        """

        X_hidden, gt_labels = [], []

        self.eval()

        with torch.no_grad():
            for x, y in train_dataset_producer:
                x, y = utils.to_tensor(x.double(), y.long(), self.device)

                logits, x_hidden = self.forward_f(x)

                X_hidden.append(x_hidden)
                gt_labels.append(y)

                _, count = torch.unique(
                    torch.cat(gt_labels), return_counts=True)
                if torch.min(count) >= self.n_centers:
                    break

            X_hidden = torch.vstack(X_hidden)
            gt_labels = torch.cat(gt_labels)

            self.gaussian_means = [X_hidden[gt_labels == i]
                                   [:self.n_centers] for i in range(self.n_classes)]

        self.get_threshold(val_dataset_producer)

        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

        self.save_to_disk()

    def load(self):
        """
        Load saved model parameters and other important data from disk.
        """
        ckpt = torch.load(self.model_save_path)
        self.gaussian_means = ckpt['gaussian_means']
        self.tau = ckpt['tau']
        self.model.load_state_dict(ckpt['base_model'])

    def save_to_disk(self):
        """
        Save model parameters and other important data to disk.
        """
        torch.save({
            'gaussian_means': self.gaussian_means,
            'tau': self.tau,
            'base_model': self.model.state_dict()
        },
            self.model_save_path
        )
