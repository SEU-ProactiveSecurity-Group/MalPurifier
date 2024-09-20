# Ensure consistent behavior in Python 2 and 3
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
logger = logging.getLogger('core.defense.fcn')

# Add an error handler to the logger
logger.addHandler(ErrorHandler)


class MalwareDetectionFCN(nn.Module):
    def __init__(self, input_size, n_classes=2, device='cpu', name='FCN', **kwargs):
        super(MalwareDetectionFCN, self).__init__()
        self.device = device
        self.name = name
        self.n_classes = n_classes

        self.parse_args(**kwargs)

        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], self.hidden_units[2])
        self.classifier = nn.Linear(self.hidden_units[2], self.n_classes)

        # Define model save path
        self.model_save_path = path.join(config.get('experiments', 'md_fcn') + '_' + self.name,
                                         'model.pth')

        # Log model architecture
        logger.info(
            '========================================fcn model architecture===============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self,
                   hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
                   smooth=False,
                   **kwargs
                   ):
        """
        Parse and set network hyperparameters.
        """

        # Use default configuration if hidden units not specified
        if hidden_units is None:
            self.hidden_units = [512, 256, 128]
        else:
            self.hidden_units = hidden_units

        # Set dropout, alpha and smooth parameters
        self.dropout = nn.Dropout(dropout)
        self.alpha_ = alpha_
        self.smooth = smooth

        # Get proc_number from kwargs
        self.proc_number = kwargs.get('proc_number', None)

        # Log warning for unknown parameters
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=-1)  # Calculate probabilities for two classes
        return probabilities

    def inference(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for generating test data

        Returns
        ----------
        Returns predicted confidences and true labels
        """
        confidences = []    # Store predicted confidences for each batch
        gt_labels = []      # Store true labels for each batch
        self.eval()         # Set model to evaluation mode

        # Use torch.no_grad() to tell PyTorch not to calculate gradients during inference
        with torch.no_grad():
            # Iterate through each batch of test data
            for x, y in test_data_producer:
                # Move data to specified device (CPU or GPU) and adjust data type
                x, y = utils.to_device(x.double(), y.long(), self.device)
                # Get logits for each batch
                logits = self.forward(x)
                # Use softmax function to get confidences for each batch, and add to confidences list
                confidences.append(F.softmax(logits, dim=-1))
                # Add true labels for each batch to gt_labels list
                gt_labels.append(y)

        # Vertically stack all batch confidences into a tensor
        confidences = torch.vstack(confidences)
        # Concatenate all batch true labels into a tensor
        gt_labels = torch.cat(gt_labels, dim=0)

        return confidences, gt_labels

    def inference_dae(self, test_data_producer):
        """
        Perform model inference to get predicted confidences and true labels

        Parameters
        ----------
        @param test_data_producer: data producer or data loader for generating test data

        Returns
        ----------
        Returns predicted confidences and true labels
        """
        confidences = []    # Store predicted confidences for each batch
        gt_labels = []      # Store true labels for each batch
        self.eval()         # Set model to evaluation mode

        # Use torch.no_grad() to tell PyTorch not to calculate gradients during inference
        with torch.no_grad():
            # Iterate through each batch of test data
            for x, y in test_data_producer:
                # Move data to specified device (CPU or GPU) and adjust data type
                x, y = utils.to_device(x.double(), y.long(), self.device)
                # Get logits for each batch
                logits = self.forward(x)
                # Use softmax function to get confidences for each batch, and add to confidences list
                confidences.append(F.softmax(logits, dim=-1))
                # Add true labels for each batch to gt_labels list
                gt_labels.append(y)

        return confidences, gt_labels

    def inference_batch_wise(self, x):
        """
        Batch-wise inference supporting only malware samples

        Parameters
        ----------
        @param x: tensor of input data

        Returns
        ----------
        Returns inference confidences and labels
        """
        # Ensure input is a tensor
        assert isinstance(x, torch.Tensor)

        # Get model output
        logit = self.forward(x)

        # Return confidences for each sample and an array of ones (representing malware samples) with same shape as logit
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def predict(self, test_data_producer, indicator_masking=True):
        """
        Predict labels and perform evaluation

        Parameters
        --------
        @param test_data_producer: torch.DataLoader, data loader for generating test data
        """
        # Perform evaluation
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1).cpu().numpy()  # Predicted labels
        y_true = y_true.cpu().numpy()                # True labels

        # Use sklearn's evaluation metrics
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)

        MSG = "The accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(accuracy * 100))

        MSG = "The balanced accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(b_accuracy * 100))

        # Check if any class is absent in the data
        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)                        # Calculate False Positive Rate
        fnr = fn / float(tp + fn)                        # Calculate False Negative Rate
        f1 = f1_score(y_true, y_pred, average='binary')  # Calculate F1 score

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%、False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    def customize_loss(self, logits, gt_labels, representation=None, mini_batch_idx=None):
        """
        Customize loss function

        Parameters
        --------
        @param logits: Tensor, model output
        @param gt_labels: Tensor, ground truth labels
        @param representation: Tensor, optional parameter, feature representation
        @param mini_batch_idx: Int, optional parameter, index of mini-batch

        Returns
        --------
        Returns cross-entropy loss
        """
        return F.cross_entropy(logits, gt_labels)

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        Train malware detector, select best model based on cross-entropy loss on validation set.

        Parameters
        ----------
        @param train_data_producer: object, iterator for generating batches of training data
        @param validation_data_producer: object, iterator for generating validation data
        @param epochs: int, number of training epochs
        @param lr: float, learning rate for Adam optimizer
        @param weight_decay: float, penalty factor
        @param verbose: bool, whether to display detailed logs
        """
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        best_avg_acc = 0.   # Record best accuracy on validation set
        best_epoch = 0      # Record epoch corresponding to best accuracy
        total_time = 0.     # Total training time

        # Get number of training data batches
        nbatches = len(train_data_producer)

        # Perform specified number of training epochs
        for i in range(epochs):
            # Set model to training mode
            self.train()
            # Initialize lists to store loss values and accuracies for each batch
            losses, accuracies = [], []

            # Iterate through each training data batch
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # Move data to specified computation device (e.g., GPU or CPU)
                x_train, y_train = utils.to_device(
                    x_train.double(), y_train.long(), self.device)

                # Record start time of training
                start_time = time.time()

                # Clear previously accumulated gradients
                optimizer.zero_grad()

                # Perform forward pass on input data
                logits = self.forward(x_train)

                # Calculate loss based on model output and true labels
                loss_train = self.customize_loss(logits, y_train)

                # Perform backward pass on loss
                loss_train.backward()

                # Update model parameters using optimizer
                optimizer.step()

                # Calculate total time spent on training this batch
                total_time += time.time() - start_time

                # Calculate accuracy on this batch
                acc_train = (logits.argmax(1) == y_train).sum(
                ).item() / x_train.size()[0]

                # Convert time to minutes and seconds
                mins, secs = int(total_time / 60), int(total_time % 60)

                # Add loss and accuracy of this batch to lists
                losses.append(loss_train.item())
                accuracies.append(acc_train)

                # If verbose mode is on, display current training progress and loss and accuracy on this batch
                if verbose:
                    logger.info(
                        f'Batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | Training time: {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Training accuracy: {acc_train * 100:.2f}')

            self.eval()  # Set model to evaluation mode
            avg_acc_val = []

            with torch.no_grad():  # Ensure no gradient calculation in evaluation mode
                for x_val, y_val in validation_data_producer:
                    # Move data to specified device (e.g., GPU or CPU), and ensure data types are double precision float and long integer
                    x_val, y_val = utils.to_device(
                        x_val.double(), y_val.long(), self.device)

                    # Perform forward pass using the model to get output results
                    logits = self.forward(x_val)

                    # Calculate accuracy on validation data
                    acc_val = (logits.argmax(1) == y_val).sum(
                    ).item() / x_val.size()[0]

                    # Save accuracy for each batch of validation data
                    avg_acc_val.append(acc_val)

                # Calculate average accuracy across all validation data
                avg_acc_val = np.mean(avg_acc_val)

            # If current epoch's validation accuracy exceeds previous best validation accuracy
            if avg_acc_val >= best_avg_acc:
                # Update best validation accuracy
                best_avg_acc = avg_acc_val
                best_epoch = i

                # Check if model save path exists, if not, create it
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))

                # Save current model parameters
                torch.save(self.state_dict(), self.model_save_path)

                # If verbose mode is on, display model save path
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            # If verbose mode is on, display training loss, training accuracy, validation accuracy, and best validation accuracy
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
