from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
import os.path as path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import IntegratedGradients

import numpy as np

from core.defense.md_dnn import MalwareDetectionDNN
from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_input_convex_nn')
logger.addHandler(ErrorHandler)

# Define an advanced malware detector using Input Convex Neural Network (ICNN)
# This code defines an advanced malware detector class that uses an Input Convex Neural Network (ICNN) structure.
# Its main goal is to provide a high-level model capable of detecting malware and optimizing its internal neural network structure.


class AdvMalwareDetectorICNN(nn.Module, DetectorTemplate):
    # Initialization function
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.98,
                 device='cpu', name='', **kwargs):
        # Call parent class initialization functions
        nn.Module.__init__(self)
        DetectorTemplate.__init__(self)

        # Set attributes such as input size, number of classes, ratio, device, and name
        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = 0.98
        self.device = device
        self.name = name
        self.parse_args(**kwargs)

        # Check if md_nn_model is an instance of nn.Module
        if isinstance(md_nn_model, nn.Module):
            self.md_nn_model = md_nn_model
        else:
            kwargs['smooth'] = True
            # If not, build a default DNN malware detection model
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            # Warn the user that a custom NN-based malware detector is being used
            warnings.warn("Use a self-defined NN-based malware detector")

        # Check if the model has a 'smooth' attribute
        if hasattr(self.md_nn_model, 'smooth'):
            # If the model is not smooth, replace ReLU with SELU
            if not self.md_nn_model.smooth:
                for name, child in self.md_nn_model.named_children():
                    if isinstance(child, nn.ReLU):
                        self.md_nn_model._modules['relu'] = nn.SELU()
        else:
            # If there's no 'smooth' attribute, replace ReLU with SELU
            for name, child in self.md_nn_model.named_children():
                if isinstance(child, nn.ReLU):
                    self.md_nn_model._modules['relu'] = nn.SELU()

        # Move the model to the specified device
        self.md_nn_model = self.md_nn_model.to(self.device)

        # Input convex neural network
        self.non_neg_dense_layers = []

        # At least one hidden layer is required
        if len(self.dense_hidden_units) < 1:
            raise ValueError("Expect at least one hidden layer.")

        # Create non-negative dense layers
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[i],
                                                       self.dense_hidden_units[i + 1],
                                                       bias=False))
        self.non_neg_dense_layers.append(
            nn.Linear(self.dense_hidden_units[-1], 1, bias=False))

        # Register non-negative dense layers
        for idx_i, dense_layer in enumerate(self.non_neg_dense_layers):
            self.add_module('non_neg_layer_{}'.format(idx_i), dense_layer)

        # Create dense layers
        self.dense_layers = []
        self.dense_layers.append(
            nn.Linear(self.input_size, self.dense_hidden_units[0]))
        for i in range(len(self.dense_hidden_units[1:])):
            self.dense_layers.append(
                nn.Linear(self.input_size, self.dense_hidden_units[i]))
        self.dense_layers.append(nn.Linear(self.input_size, 1))

        # Register dense layers
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('layer_{}'.format(idx_i), dense_layer)

        # Create parameter tau and set it to not require gradients
        self.tau = nn.Parameter(torch.zeros(
            [1, ], device=self.device), requires_grad=False)

        # Set the model's save path
        self.model_save_path = path.join(config.get('experiments', 'amd_icnn') + '_' + self.name,
                                         'model.pth')
        # Print the model's structure information
        logger.info(
            '========================================icnn model architecture==============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

    def parse_args(self,
                   dense_hidden_units=None,  # List of hidden units for dense layers
                   dropout=0.6,               # Dropout rate
                   alpha_=0.2,                # Alpha parameter
                   **kwargs                   # Other keyword arguments
                   ):
        # If no dense hidden units are provided, use default [200, 200]
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]

        # If the provided dense hidden units are in list form, assign directly
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units

        # If not provided as a list, raise a TypeError
        else:
            raise TypeError("Expect a list of hidden units.")

        # Set dropout rate
        self.dropout = dropout
        # Set alpha parameter
        self.alpha_ = alpha_
        # Get the `proc_number` parameter
        self.proc_number = kwargs['proc_number']
        # If additional keyword arguments are provided and the number of arguments is greater than 0, log a warning message
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    # Define forward_f function, which applies the md_nn_model to input x
    def forward_f(self, x):
        return self.md_nn_model(x)

    # Define forward_g function, which processes input data x and passes it to dense layers and non-negative dense layers
    def forward_g(self, x):
        # Initialize prev_x as None, used to store the value of the previous x
        prev_x = None
        # Enumerate each dense layer
        for i, dense_layer in enumerate(self.dense_layers):
            # Initialize x_add list to store intermediate results
            x_add = []

            # Pass input x through the current dense layer
            x1 = dense_layer(x)

            # Add the result to the x_add list
            x_add.append(x1)

            # If prev_x is not None, it means it's not the first dense layer
            if prev_x is not None:
                # Pass the previous x through the non-negative dense layer
                x2 = self.non_neg_dense_layers[i - 1](prev_x)
                # Add the result to the x_add list
                x_add.append(x2)

            # Sum all elements in the x_add list
            prev_x = torch.sum(torch.stack(x_add, dim=0), dim=0)

            # If it's not the last dense layer, apply the SELU activation function
            if i < len(self.dense_layers):
                prev_x = F.selu(prev_x)

        # Change the shape of the output and return
        return prev_x.reshape(-1)

    def forward(self, x):
        return self.forward_f(x), self.forward_g(x)

    # Define the forward propagation function
    def forward(self, x):
        # Pass input x to both forward_f and forward_g functions simultaneously
        return self.forward_f(x), self.forward_g(x)

    # Define the prediction function
    def predict(self, test_data_producer, indicator_masking=True):
        """
        Predict labels and evaluate the detector and indicator

        Parameters:
        --------
        @param test_data_producer: torch.DataLoader, for producing test data
        @param indicator_masking: whether to filter low-density examples or mask their values
        """
        # Perform inference from the test data generator to get central prediction values, probabilities, and true labels
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        # Get the maximum index of the prediction as the prediction result
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        # Calculate the indicator flag
        indicator_flag = self.indicator(x_prob).cpu().numpy()

        # Define the evaluation function
        def measurement(_y_true, _y_pred):
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            # Calculate and print accuracy
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info("Accuracy on test dataset: {:.5f}%".format(accuracy * 100))
            logger.info("Balanced accuracy on test dataset: {:.5f}%".format(b_accuracy * 100))
            # Check if a certain class is completely missing
            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("Some classes are missing.")
                return

            # Calculate confusion matrix and get TP, TN, FP, FN
            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            # Calculate F1 score
            f1 = f1_score(_y_true, _y_pred, average='binary')
            logger.info("False negative rate: {:.5f}%, false positive rate: {:.5f}%, F1-score: {:.5f}%".format(
                fnr * 100, fpr * 100, f1 * 100))

        # Evaluate true labels and predicted labels
        measurement(y_true, y_pred)

        rtn_value = (y_pred == 0) & indicator_flag

        if indicator_masking:
            # Exclude samples with "uncertain" responses
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # Here, instead of filtering out examples, reset predictions to 1
            y_pred[~indicator_flag] = 1.
        logger.info('Indicator is enabled...')
        logger.info('Threshold: {:.5}'.format(self.tau.item()))
        # Evaluate again
        measurement(y_true, y_pred)

        return rtn_value

    # Define the inference function
    def inference(self, test_data_producer):
        # Initialize three empty lists: y_cent for storing predicted class center values, x_prob for storing predicted probability values, gt_labels for storing true labels.
        y_cent, x_prob = [], []
        gt_labels = []

        # Set the model to evaluation mode
        self.eval()

        # Use torch.no_grad() to indicate that PyTorch should not compute gradients in this context, which is common practice during inference to save memory and speed up computation.
        with torch.no_grad():
            # Iterate through each batch of data in the test data generator
            for x, y in test_data_producer:
                # Move data to the device and ensure x's data type is double and y's data type is long
                x, y = utils.to_device(x.double(), y.long(), self.device)

                # Get logits_f and logits_g through forward propagation
                logits_f, logits_g = self.forward(x)

                # Use softmax function to calculate the probability distribution of logits_f and add it to the y_cent list
                y_cent.append(torch.softmax(logits_f, dim=-1))

                # Add logits_g to the x_prob list
                x_prob.append(logits_g)

                # Add true labels to the gt_labels list
                gt_labels.append(y)

        # Use torch.cat to concatenate all Tensors in the three lists along dimension 0
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)

        # Return three Tensors: y_cent, x_prob, gt_labels
        return y_cent, x_prob, gt_labels

    # This code's main purpose is to calculate the importance or contribution of model inputs.
    # Integrated gradients is a method for explaining machine learning models, providing a way to understand how each input feature contributes to the prediction result.
    # Here, this method is used for two different model outputs: the classification task (forward_f) and another output that may be related to density estimation or some specific task (forward_g).
    def get_important_attributes(self, test_data_producer, indicator_masking=False):
        """
        Get important attributes of inputs using integrated gradients method.

        The adjacency matrix will be ignored.
        """
        # Store attribute importance for the classification task
        attributions_cls = []
        # Store attribute importance for other tasks (possibly density estimation or some task)
        attributions_de = []

        # Define a wrapper function for integrated gradient calculation for the classification task
        def _ig_wrapper_cls(_x):
            logits = self.forward_f(_x)  # Get model predictions for input x
            return F.softmax(logits, dim=-1)  # Apply softmax to predictions to get probability values

        # Initialize integrated gradients method for the classification task
        ig_cls = IntegratedGradients(_ig_wrapper_cls)

        # Define a wrapper function for integrated gradient calculation for other tasks
        def _ig_wrapper_de(_x):
            return self.forward_g(_x)

        # Initialize integrated gradients method for other tasks
        ig_de = IntegratedGradients(_ig_wrapper_de)

        # Iterate through test data
        for i, (x, y) in enumerate(test_data_producer):
            x, y = utils.to_tensor(x, y, self.device)  # Convert input and labels to tensors
            x.requires_grad = True  # Set gradient property for input x to allow subsequent gradient calculation
            base_lines = torch.zeros_like(
                x, dtype=torch.double, device=self.device)  # Set baseline to all zeros
            base_lines[:, -1] = 1  # Modify the last value of the baseline to 1
            # Calculate attribute importance for the classification task
            attribution_bs = ig_cls.attribute(x,
                                              baselines=base_lines,
                                              target=1)  # target=1 means we calculate attribute importance for class 1
            attributions_cls.append(
                attribution_bs.clone().detach().cpu().numpy())

            # Calculate attribute importance for other tasks
            attribution_bs = ig_de.attribute(x,
                                             baselines=base_lines
                                             )
            attributions_de.append(
                attribution_bs.clone().detach().cpu().numpy())

        # Combine results from all batches into one array
        return np.vstack(attributions_cls), np.vstack(attributions_de)

    def inference_batch_wise(self, x):
        """
        Return classification probabilities and g model output.
        """
        assert isinstance(x, torch.Tensor)  # Assert to ensure input is of torch.Tensor type
        self.eval()  # Set the model to evaluation mode
        logits_f, logits_g = self.forward(x)  # Get outputs from f and g models
        # Apply softmax to f model output to get classification probabilities, and move results to CPU
        return torch.softmax(logits_f, dim=-1).detach().cpu().numpy(), logits_g.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau  # Return tau, i.e., the decision threshold

    def indicator(self, x_prob, y_pred=None):
        """
        Determine if a sample is original.
        """
        if isinstance(x_prob, np.ndarray):  # Check if input is a numpy array
            # Convert numpy array to torch.Tensor
            x_prob = torch.tensor(x_prob, device=self.device)
            # Determine if each sample's probability is less than or equal to tau, and return the result
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):  # Check if input is a torch.Tensor
            return x_prob <= self.tau  # Determine if each sample's probability is less than or equal to tau, and return the result
        else:
            # If input is neither a numpy array nor a torch.Tensor, raise a TypeError
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # In short, this method calculates the model's output probabilities, sorts these probabilities, and then determines a threshold based on the provided ratio.
    # When the model's output is below this threshold, the model will consider the input as adversarial.
    def get_threshold(self, validation_data_producer, ratio=None):
        """
        Get the threshold for adversarial detection.

        Parameters:
        --------
        validation_data_producer : Object
            An iterator for generating validation dataset.
        ratio : float, optional
            The ratio used to calculate the threshold, default is self.ratio.

        """
        self.eval()  # Set the model to evaluation mode
        # If ratio is not provided, use self.ratio as the default value
        ratio = ratio if ratio is not None else self.ratio

        # Assert to ensure the ratio value is in the range [0,1]
        assert 0 <= ratio <= 1
        probabilities = []  # For storing model output probability values
        with torch.no_grad():  # Without computing gradients
            for x_val, y_val in validation_data_producer:  # Get data from the validation data generator
                # Convert input data and labels to appropriate data types and move to the specified device
                x_val, y_val = utils.to_tensor(
                    x_val.double(), y_val.long(), self.device)
                # Get g model output
                x_logits = self.forward_g(x_val)
                # Add model output to the probability list
                probabilities.append(x_logits)
            # Sort all model outputs
            s, _ = torch.sort(torch.cat(probabilities, dim=0))
            # Calculate index i, which determines the threshold position in the sorted output based on the provided ratio
            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0  # Ensure i is a valid index
            # Set the model's threshold tau to s[i], i.e., the threshold determined by the ratio
            self.tau[0] = s[i]

    def reset_threshold(self):
        """
        Reset the model's threshold to 0.
        """
        self.tau[0] = 0.

    # This custom loss function aims to train the model to accurately classify original samples while detecting adversarial samples. This is achieved by combining two types of losses, each with its own weight.
    def customize_loss(self, logits_x, labels, logits_adv_x, labels_adv, beta_1=1, beta_2=1):
        """
        Custom loss function combining classification loss and adversarial loss.

        Parameters:
        --------
        logits_x : torch.Tensor
            Model output for original samples.
        labels : torch.Tensor
            True labels for original samples.
        logits_adv_x : torch.Tensor
            Model output for adversarial samples.
        labels_adv : torch.Tensor
            True labels for adversarial samples.
        beta_1 : float, optional
            Weight for original sample loss, default is 1.
        beta_2 : float, optional
            Weight for adversarial sample loss, default is 1.

        Returns:
        --------
        torch.Tensor
            Calculated total loss.

        """
        # If there are adversarial samples, calculate adversarial loss. Otherwise, set it to 0.
        if logits_adv_x is not None and len(logits_adv_x) > 0:
            G = F.binary_cross_entropy_with_logits(logits_adv_x, labels_adv)
        else:
            G = 0

        # If there are original samples, calculate classification loss. Otherwise, set it to 0.
        if logits_x is not None and len(logits_x) > 0:
            F_ = F.cross_entropy(logits_x, labels)
        else:
            F_ = 0

        # Combine the two losses, using beta_1 and beta_2 as weights
        return beta_1 * F_ + beta_2 * G

    # This code describes the training process. It first sets the model to training mode at the beginning of each epoch, then trains on each batch of data.
    # A highlight here is that it also generates data with salt and pepper noise and classifies it.
    # Finally, it calculates the loss and accuracy for each batch and may log them.

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., verbose=True):
        """
        Train the malware and adversarial detector, select the best model based on validation results.

        Parameters:
        --------
        train_data_producer: object
            An iterator for generating training batch data.
        validation_data_producer: object
            An iterator for generating validation dataset.
        epochs: int
            Number of training iterations, default is 100.
        lr: float
            Learning rate for Adam optimizer, default is 0.005.
        weight_decay: float
            Penalty factor, default is 0.
        verbose: bool
            Whether to display detailed logs, default is True.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        best_avg_acc = 0.  # Initialize best average accuracy
        best_epoch = 0
        total_time = 0.  # Cumulative training time
        nbatches = len(train_data_producer)

        # Start training
        for i in range(epochs):
            self.train()  # Set the model to training mode
            losses, accuracies = [], []

            # Iterate through training batch data
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                # Move data to specified device
                x_train, y_train = utils.to_device(
                    x_train.double(), y_train.long(), self.device)

                # Generate data for g network
                # 1. Add salt and pepper noise
                x_train_noises = torch.clamp(
                    x_train + utils.psn(x_train, np.random.uniform(0, 0.5)), min=0., max=1.)
                x_train_ = torch.cat([x_train, x_train_noises], dim=0)
                y_train_ = torch.cat([torch.zeros(x_train.shape[:1]), torch.ones(
                    x_train.shape[:1])]).double().to(self.device)
                idx = torch.randperm(y_train_.shape[0])
                x_train_ = x_train_[idx]
                y_train_ = y_train_[idx]

                # Start a training iteration
                start_time = time.time()
                optimizer.zero_grad()
                logits_f = self.forward_f(x_train)
                logits_g = self.forward_g(x_train_)
                loss_train = self.customize_loss(
                    logits_f, y_train, logits_g, y_train_)
                loss_train.backward()
                optimizer.step()

                # Constraints
                constraint = utils.NonnegWeightConstraint()
                for name, module in self.named_modules():
                    if 'non_neg_layer' in name:
                        module.apply(constraint)

                total_time = total_time + time.time() - start_time

                # Calculate accuracy
                acc_f_train = (logits_f.argmax(
                    1) == y_train).sum().item() / x_train.size()[0]
                acc_g_train = ((F.sigmoid(logits_g) >= 0.5) ==
                               y_train_).sum().item() / x_train_.size()[0]

                # Update records
                losses.append(loss_train.item())
                accuracies.append(acc_f_train)
                accuracies.append(acc_g_train)

                # If needed, print detailed logs
                if verbose:
                    mins, secs = int(total_time / 60), int(total_time % 60)
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}% & {acc_g_train * 100:.2f}%.')

            # Set the model to evaluation mode
            self.eval()

            # Initialize a list to store accuracy for each batch of validation data
            avg_acc_val = []

            # Disable gradient calculation to speed up computation and reduce memory usage
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    # Move data to specified device
                    x_val, y_val = utils.to_device(
                        x_val.double(), y_val.long(), self.device)

                    # Generate data for g network (with salt and pepper noise)
                    x_val_noises = torch.clamp(
                        x_val + utils.psn(x_val, np.random.uniform(0, 0.5)), min=0., max=1.)
                    x_val_ = torch.cat([x_val, x_val_noises], dim=0)
                    y_val_ = torch.cat([torch.zeros(x_val.shape[:1]), torch.ones(
                        x_val.shape[:1])]).long().to(self.device)

                    # Get predicted labels
                    logits_f = self.forward_f(x_val)
                    logits_g = self.forward_g(x_val_)

                    # Calculate accuracy for f network
                    acc_val = (logits_f.argmax(1) ==
                               y_val).sum().item() / x_val.size()[0]
                    avg_acc_val.append(acc_val)

                    # Calculate accuracy for g network
                    acc_val_g = ((F.sigmoid(logits_g) >= 0.5) ==
                                 y_val_).sum().item() / x_val_.size()[0]
                    avg_acc_val.append(acc_val_g)

                # Calculate average accuracy
                avg_acc_val = np.mean(avg_acc_val)

            # If the current model's validation accuracy is the best so far, save this model
            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                # Get threshold
                self.get_threshold(validation_data_producer)
                # Save model
                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            # If needed, display detailed information about training and validation
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        # load model
        assert path.exists(self.model_save_path), 'train model first'
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        torch.save(self.state_dict(), self.model_save_path)
