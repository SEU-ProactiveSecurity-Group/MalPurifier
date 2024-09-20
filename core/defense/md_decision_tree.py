from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from captum.attr import IntegratedGradients

import numpy as np

from config import config, logging, ErrorHandler

from tools import utils

logger = logging.getLogger('core.defense.dt')
logger.addHandler(ErrorHandler)


class MalwareDetectionDT:
    def __init__(self, max_depth=None, random_state=0, name='DT', device='cpu', **kwargs):
        """
        Initialize malware detector

        Parameters:
        ----------
        @param max_depth: Maximum depth of the tree.
        @param random_state: Integer, random state for reproducibility.
        @param name: String, name of the model.
        @param device: Device to run the model on.
        """

        del kwargs

        self.max_depth = max_depth
        self.random_state = random_state
        self.name = name
        self.n_classes = 2  # binary classification
        self.device = device

        self.model = DecisionTreeClassifier(max_depth=self.max_depth,
                                            random_state=self.random_state)

        self.scaler = StandardScaler()  # for data standardization

        # Define model save path
        self.model_save_path = path.join(config.get('experiments', 'dt') + '_' + self.name,
                                         'model.pkl')

    def forward(self, x):
        """
        Get prediction confidences from the decision tree model

        Parameters
        ----------
        @param x: 2D tensor or array, feature representation

        Returns
        ----------
        Prediction confidences
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x = self.scaler.transform(x)

        confidences = self.model.predict_proba(x)

        confidences = torch.tensor(confidences, dtype=torch.float32)

        return confidences

    def cross_validate(self, X, y, cv=5):
        """
        Evaluate model performance using cross-validation.

        Parameters:
        ----------
        @param X: Training data.
        @param y: Target labels.
        @param cv: Number of cross-validation folds.

        Returns:
        ----------
        scores: List of cross-validation scores.
        """
        X = self.scaler.fit_transform(X)

        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores

    def inference(self, test_data_producer):
        """
        Perform model inference to get prediction confidences and true labels

        Parameters
        ----------
        @param test_data_producer: Data producer or loader for test data

        Returns
        ----------
        Prediction confidences and true labels
        """
        confidences = []
        gt_labels = []

        for x, y in test_data_producer:
            x = self.scaler.transform(x.numpy())
            confidence_batch = self.model.predict_proba(x)
            confidences.append(confidence_batch)
            gt_labels.append(y.numpy())

        confidences = np.vstack(confidences)
        gt_labels = np.hstack(gt_labels)

        return confidences, gt_labels

    def inference_dae(self, test_data_producer):
        """
        Perform model inference to get prediction confidences and true labels

        Parameters
        ----------
        @param test_data_producer: Data producer or loader for test data

        Returns
        ----------
        Prediction confidences and true labels
        """
        confidences = []
        gt_labels = []

        for x, y in test_data_producer:
            x = self.scaler.transform(x.numpy())
            confidence_batch = self.model.predict_proba(x)
            confidences.append(confidence_batch)
            gt_labels.append(y.numpy())

        return confidences, gt_labels

    def fit(self, train_data_producer, val_data_producer, early_stopping_rounds=30, n_resamples=None):
        # Load validation data
        all_X_val = []
        all_y_val = []

        for batch_data in tqdm(val_data_producer, desc="Loading validation data"):
            X_batch, y_batch = batch_data
            all_X_val.append(X_batch.numpy())
            all_y_val.append(y_batch.numpy())

        X_val = np.vstack(all_X_val)
        y_val = np.hstack(all_y_val)

        X_val = self.scaler.fit_transform(X_val)

        best_val_accuracy = 0
        no_improve_rounds = 0

        n_samples_per_batch = n_resamples or len(y_val)

        for epoch, batch_data in enumerate(tqdm(train_data_producer, desc="Training batches")):
            X_batch, y_batch = batch_data
            X_batch = X_batch.numpy()
            y_batch = y_batch.numpy()

            indices = np.random.choice(
                len(y_batch), n_samples_per_batch, replace=True)
            X_resampled = X_batch[indices]
            y_resampled = y_batch[indices]

            X_resampled = self.scaler.transform(X_resampled)

            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=self.random_state)
            self.model.fit(X_resampled, y_resampled)

            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= early_stopping_rounds:
                logger.info(
                    f"Early stopping triggered as accuracy hasn't improved for {early_stopping_rounds} rounds.")
                break

        logger.info(
            f'{self.name} model trained and validated with best accuracy: {best_val_accuracy}')

    def inference_batch_wise(self, x):
        """
        Batch-wise inference for malware samples only

        Parameters
        ----------
        @param x: Input data tensor

        Returns
        ----------
        Inference confidences and labels
        """
        assert isinstance(x, torch.Tensor)

        x = x.detach().cpu().numpy()

        confidences = self.forward(x)

        return confidences, np.ones((confidences.size()[0],))

    def predict(self, test_data_producer, indicator_masking=True):
        """
        Predict labels and evaluate

        Parameters
        --------
        @param test_data_producer: torch.DataLoader, data loader for test data
        """
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1)

        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)

        logger.info(f"The accuracy on the test dataset is {accuracy * 100:.5f}%")
        logger.info(f"The balanced accuracy on the test dataset is {b_accuracy * 100:.5f}%")

        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        logger.info(f"False Negative Rate (FNR) is {fnr * 100:.5f}%, False Positive Rate (FPR) is {fpr * 100:.5f}%, F1 score is {f1 * 100:.5f}%")

    def save_model(self):
        """
        Save current model to disk.
        """
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

        with open(self.model_save_path, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)
        logger.info(f'Model saved to {self.model_save_path}')

    def eval(self):
        pass

    def load(self):
        """
        Load model from disk.
        """
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, 'rb') as file:
                saved = pickle.load(file)
                self.model = saved['model']
                self.scaler = saved['scaler']
            logger.info(f'Model loaded from {self.model_save_path}')
        else:
            logger.error(f'Model file not found at {self.model_save_path}')

    def load_state_dict(self):
        self.load()
