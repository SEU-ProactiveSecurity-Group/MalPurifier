# Import future features to ensure consistent behavior in Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import basic libraries
import time
import os.path as path

# Import PyTorch related libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import Captum library for model interpretability
from captum.attr import IntegratedGradients

# Import NumPy
import numpy as np

# Import configuration, logging, and error handling from config module
from config import config, logging, ErrorHandler

# Import custom tools module
from tools import utils

# Initialize logger and set its name
logger = logging.getLogger('core.defense.dnn')

# Add an error handler to the logger
logger.addHandler(ErrorHandler)


# Define the Malware Detection Deep Neural Network
class MalwareDetectionDNN(nn.Module):
    def __init__(self, input_size, n_classes, device='cpu', name='DNN', **kwargs):
        """
        Initialize the malware detector

        Parameters:
        ----------
        @param input_size: int, number of input features.
        @param n_classes: int, number of classes for classification.
        @param device: str, 'cpu' or 'cuda' to indicate where the model should run.
        @param name: str, name of the model.
        """
        super(MalwareDetectionDNN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        self.name = name

        self.parse_args(**kwargs)

        self.dense_layers = []

        if len(self.dense_hidden_units) >= 1:
            self.dense_layers.append(
                nn.Linear(self.input_size, self.dense_hidden_units[0]))
        else:
            raise ValueError("Expect at least one hidden layer.")

        for i in range(len(self.dense_hidden_units[0:-1])):
            self.dense_layers.append(nn.Linear(self.dense_hidden_units[i],
                                               self.dense_hidden_units[i + 1]))

        self.dense_layers.append(
            nn.Linear(self.dense_hidden_units[-1], self.n_classes))

        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('nn_model_layer_{}'.format(idx_i), dense_layer)

        self.activation_func = F.selu if self.smooth else F.relu

        self.model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + self.name,
                                         'model.pth')

        logger.info(
            '========================================dnn model architecture===============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self,
                   dense_hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
                   smooth=False,
                   **kwargs
                   ):
        """
        Parse and set network hyperparameters.

        Parameters:
        ----------
        dense_hidden_units : list, optional
            Number of units in each hidden layer.
        dropout : float, optional
            Dropout rate for regularization.
        alpha_ : float, optional
            Parameter for certain activation functions.
        smooth : bool, optional
            Whether to use smooth activation function.
        **kwargs : dict
            Other hyperparameters.
        """

        self.dense_hidden_units = [200, 200] if dense_hidden_units is None else dense_hidden_units
        if not isinstance(self.dense_hidden_units, list):
            raise TypeError("Expect a list of hidden units.")

        self.dropout = dropout
        self.alpha_ = alpha_
        self.smooth = smooth

        self.proc_number = kwargs.get('proc_number', None)

        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        Forward pass through the neural network

        Parameters
        ----------
        @param x: 2D tensor, feature representation
        """
        for dense_layer in self.dense_layers[:-1]:
            x = self.activation_func(dense_layer(x))

        latent_representation = F.dropout(
            x, self.dropout, training=self.training)

        logits = self.dense_layers[-1](latent_representation)
        return logits

    def inference(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for test data

        Returns
        ----------
        Predicted confidences and true labels
        """
        confidences = []
        gt_labels = []
        self.eval()

        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits = self.forward(x)
                confidences.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)

        confidences = torch.vstack(confidences)
        gt_labels = torch.cat(gt_labels, dim=0)

        return confidences, gt_labels

    def inference_dae(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for test data

        Returns
        ----------
        Predicted confidences and true labels
        """
        confidences = []
        gt_labels = []
        self.eval()

        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits = self.forward(x)
                confidences.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)

        return confidences, gt_labels

    def get_important_attributes(self, test_data_producer, target_label=1):
        """
        Get important attributes/features using Integrated Gradients method

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for test data
        @param target_label: target label, default is 1

        Returns
        ----------
        Important attributes/features
        """
        attributions = []
        gt_labels = []

        def _ig_wrapper(_x):
            logits = self.forward(_x)
            return F.softmax(logits, dim=-1)

        ig = IntegratedGradients(_ig_wrapper)

        for i, (x, y) in enumerate(test_data_producer):
            x, y = utils.to_device(x.double(), y.long(), self.device)
            x.requires_grad = True
            baseline = torch.zeros_like(
                x, dtype=torch.double, device=self.device)
            attribution_bs = ig.attribute(x,
                                          baselines=baseline,
                                          target=target_label)
            attribution = torch.hstack(attribution_bs)
            attributions.append(attribution.clone().detach().cpu().numpy())
            gt_labels.append(y.clone().detach().cpu().numpy())
            np.save('./labels', np.concatenate(gt_labels))

        return np.vstack(attributions)

    def inference_batch_wise(self, x):
        """
        Batch-wise inference for malware samples only

        Parameters
        ----------
        @param x: tensor of input data

        Returns
        ----------
        Inference confidences and labels
        """
        assert isinstance(x, torch.Tensor)

        logit = self.forward(x)

        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def predict(self, test_data_producer, indicator_masking=True):
        """
        Predict labels and evaluate

        Parameters
        --------
        @param test_data_producer: torch.DataLoader, data loader for test data
        """
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)

        logger.info("The accuracy on the test dataset is {:.5f}%".format(accuracy * 100))
        logger.info("The balanced accuracy on the test dataset is {:.5f}%".format(b_accuracy * 100))

        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        print("Other evaluation metrics we may need:")
        logger.info("False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%".format(fnr * 100, fpr * 100, f1 * 100))

    def customize_loss(self, logits, gt_labels, representation=None, mini_batch_idx=None):
        """
        Customize loss function

        Parameters
        --------
        @param logits: Tensor, model output
        @param gt_labels: Tensor, ground truth labels
        @param representation: Tensor, optional, feature representation
        @param mini_batch_idx: Int, optional, mini-batch index

        Returns
        --------
        Cross-entropy loss
        """
        return F.cross_entropy(logits, gt_labels)

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        Train the malware detector, selecting the best model based on cross-entropy loss on the validation set.

        Parameters
        ----------
        @param train_data_producer: object, iterator for generating batches of training data
        @param validation_data_producer: object, iterator for generating validation data
        @param epochs: int, number of training epochs
        @param lr: float, learning rate for Adam optimizer
        @param weight_decay: float, penalty factor
        @param verbose: bool, whether to display detailed logs
        """
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.

        nbatches = len(train_data_producer)

        for i in range(epochs):
            self.train()
            losses, accuracies = [], []

            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train, y_train = utils.to_device(
                    x_train.double(), y_train.long(), self.device)

                start_time = time.time()

                optimizer.zero_grad()

                logits = self.forward(x_train)

                loss_train = self.customize_loss(logits, y_train)

                loss_train.backward()

                optimizer.step()

                total_time += time.time() - start_time

                acc_train = (logits.argmax(1) == y_train).sum(
                ).item() / x_train.size()[0]

                mins, secs = int(total_time / 60), int(total_time % 60)

                losses.append(loss_train.item())
                accuracies.append(acc_train)

                if verbose:
                    logger.info(
                        f'Batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | Time: {mins:.0f} m, {secs} s.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Training accuracy: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []

            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val, y_val = utils.to_device(
                        x_val.double(), y_val.long(), self.device)

                    logits = self.forward(x_val)

                    acc_val = (logits.argmax(1) == y_val).sum(
                    ).item() / x_val.size()[0]

                    avg_acc_val.append(acc_val)

                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i

                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))

                torch.save(self.state_dict(), self.model_save_path)

                if verbose:
                    print(f'Model saved at: {self.model_save_path}')

            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Training accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | Best validation accuracy: {best_avg_acc * 100:.2f} at epoch {best_epoch}')

    def load(self):
        """
        Load model parameters from disk
        """
        self.load_state_dict(torch.load(self.model_save_path))
