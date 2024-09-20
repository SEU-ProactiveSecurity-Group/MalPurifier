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
logger = logging.getLogger('core.defense.lstm')

# Add an error handler to the logger
logger.addHandler(ErrorHandler)


class MalwareDetectionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, n_classes=2, seq_len=1, device='cpu', name='LSTM', **kwargs):
        """
        Initialize the malware detector

        Parameters:
        ----------
        @param input_dim: int, dimension of input features at each time step.
        @param hidden_dim: int, dimension of LSTM hidden state.
        @param n_classes: int, number of classes for classification.
        @param seq_len: int, length of the sequence.
        @param device: str, 'cpu' or 'cuda', indicating where the model should run.
        @param name: str, name of the model.
        """
        super(MalwareDetectionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.device = device
        self.name = name

        self.parse_args(**kwargs)

        # Define LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # Define output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)

        # Choose activation function based on parameters
        self.activation_func = F.selu if self.smooth else F.relu

        # Define model save path
        self.model_save_path = path.join(config.get('experiments', 'md_lstm') + '_' + self.name,
                                         'model.pth')
        # Log model structure
        logger.info(
            '======================================lstm model architecture===============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self, dropout=0.6, alpha_=0.2, smooth=False, **kwargs):
        """
        Parse and set network hyperparameters.
        """
        self.dropout = dropout
        self.alpha_ = alpha_
        self.smooth = smooth
        self.proc_number = kwargs.get('proc_number', None)

        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        Forward pass through the neural network
        """
        x = x.view(-1, self.seq_len, self.input_dim)
        lstm_out, _ = self.lstm(x)
        latent_representation = lstm_out[:, -1, :]
        latent_representation = F.dropout(
            latent_representation, self.dropout, training=self.training)
        logits = self.output_layer(latent_representation)
        return logits

    def inference(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for test data

        Returns
        ----------
        Returns predicted confidences and true labels
        """
        confidences = []
        gt_labels = []
        self.eval()

        with torch.no_grad():
            for x, y in test_data_producer:
                x = x.view(-1, self.seq_len, self.input_dim)
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits = self.forward(x)
                confidences.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)

        confidences = torch.vstack(confidences)
        gt_labels = torch.cat(gt_labels, dim=0)

        return confidences, gt_labels

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

        logger.info("Other evaluation metrics we may need:")
        logger.info("False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%".format(fnr * 100, fpr * 100, f1 * 100))

    def customize_loss(self, logits, gt_labels, representation=None, mini_batch_idx=None):
        """
        Customize loss function

        Parameters
        --------
        @param logits: Tensor, model output
        @param gt_labels: Tensor, true labels
        @param representation: Tensor, optional, feature representation
        @param mini_batch_idx: Int, optional, mini-batch index

        Returns
        --------
        Returns cross-entropy loss
        """
        return F.cross_entropy(logits, gt_labels)

    def inference_batch_wise(self, x):
        """
        Batch-wise inference for malware samples only

        Parameters
        ----------
        @param x: tensor of input data

        Returns
        ----------
        Returns inference confidences and labels
        """
        assert isinstance(x, torch.Tensor)
        x = x.view(-1, self.seq_len, self.input_dim)
        logit = self.forward(x)
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        Train the LSTM model for malware detection, selecting the best model based on cross-entropy loss on the validation set.
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
                x_train = x_train.view(-1, self.seq_len, self.input_dim)
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
                        f'Batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | Training time: {mins:.0f} min, {secs} sec.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Training accuracy: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []

            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val = x_val.view(-1, self.seq_len, self.input_dim)
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
