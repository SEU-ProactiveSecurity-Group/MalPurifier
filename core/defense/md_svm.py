# Import future features for consistent behavior in Python 2 and 3
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
from torch.optim import lr_scheduler

# Import Captum library for model interpretability
from captum.attr import IntegratedGradients

# Import NumPy
import numpy as np

# Import configuration, logging and error handling from config module
from config import config, logging, ErrorHandler

# Import custom tools module
from tools import utils

# Initialize logger and set its name
logger = logging.getLogger('core.defense.svm')

# Add an error handler to the logger
logger.addHandler(ErrorHandler)


class MalwareDetectionSVM(nn.Module):
    """
    Using fully connected neural network to implement linear SVM and Logistic regression with hinge loss and
    cross-entropy loss which computes softmax internally, respectively.
    """

    def __init__(self, input_size, n_classes=2, device='cpu', name='md_svm', **kwargs):
        super(MalwareDetectionSVM, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        self.name = name

        self.fc = nn.Linear(self.input_size, self.n_classes)

        self.parse_args(**kwargs)

        self.model_save_path = path.join(config.get('experiments', 'md_svm') + '_' + self.name,
                                         'model.pth')

        logger.info(
            '========================================svm model architecture===============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self,
                   **kwargs
                   ):
        """
        Parse and set network hyperparameters.
        """
        self.proc_number = kwargs.get('proc_number', None)

        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        out = self.fc(x)
        return torch.sigmoid(out).squeeze()

    def inference(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels.

        Parameters
        ----------
        @param test_data_producer: Data producer or data loader for test data

        Returns
        ----------
        Returns predicted confidences and true labels
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
        Perform model inference to get predicted confidences and true labels.

        Parameters
        ----------
        @param test_data_producer: Data producer or data loader for test data

        Returns
        ----------
        Returns predicted confidences and true labels
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

    def inference_batch_wise(self, x):
        """
        Batch-wise inference for malware samples only

        Parameters
        ----------
        @param x: Input data tensor

        Returns
        ----------
        Returns inference confidences and labels
        """
        assert isinstance(x, torch.Tensor)

        logit = self.forward(x)

        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        Train the SVM model and select the best model based on validation loss.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        criterion = nn.MultiMarginLoss()

        best_avg_acc = 0.
        best_epoch = 0

        nbatches = len(train_data_producer)

        for i in range(epochs):
            self.train()
            running_corrects = 0

            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train = x_train.double().to(self.device)
                y_train = y_train.long().to(self.device)

                optimizer.zero_grad()

                outputs = self.forward(x_train)
                loss = criterion(outputs, y_train)

                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == y_train).item()

                if verbose and (idx_batch % 10 == 0):
                    print(
                        f"Epoch {i}/{epochs}, Batch {idx_batch}/{nbatches} - Loss: {loss.item():.4f}")

            epoch_acc = running_corrects / len(train_data_producer.dataset)

            self.eval()
            val_corrects = 0
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val = x_val.double().to(self.device)
                    y_val = y_val.long().to(self.device)

                    outputs = self.forward(x_val)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == y_val).item()

            val_acc = val_corrects / len(validation_data_producer.dataset)

            if val_acc > best_avg_acc:
                best_avg_acc = val_acc
                best_epoch = i

                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))

                torch.save(self.state_dict(), self.model_save_path)

                if verbose:
                    print(f'Model saved at: {self.model_save_path}')

            if verbose:
                print(
                    f"Epoch {i}/{epochs} - Training Accuracy: {epoch_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        print(
            f"Best Validation Accuracy: {best_avg_acc:.4f} at Epoch {best_epoch}")

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

        MSG = "The accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(accuracy * 100))

        MSG = "The balanced accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(b_accuracy * 100))

        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    def load(self):
        """
        Load model parameters from disk
        """
        self.load_state_dict(torch.load(self.model_save_path))
