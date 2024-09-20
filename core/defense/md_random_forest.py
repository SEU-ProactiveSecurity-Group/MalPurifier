from __future__ import absolute_import
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
import os
import time
import os.path as path
import torch
import numpy as np
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.rf')
logger.addHandler(ErrorHandler)


class MalwareDetectionRF:
    def __init__(self, n_estimators=100, device='cpu', max_depth=None, random_state=0, name='RF', **kwargs):
        """
        Initialize the malware detector

        Parameters:
        ----------
        @param n_estimators: Number of trees. 
        @param max_depth: Maximum depth of trees. 
        @param random_state: Integer, random state for reproducibility.
        @param name: String, used to name the model.
        """

        del kwargs

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.name = name
        self.device = device
        self.n_classes = 2

        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state,
                                            n_jobs=-1)

        self.scaler = StandardScaler()

        self.model_save_path = path.join(config.get('experiments', 'rf') + '_' + self.name,
                                         'model.pkl')

    def eval(self):
        pass

    def forward(self, x):
        """
        Get prediction confidences through the random forest model

        Parameters
        ----------
        @param x: 2D tensor or array, feature representation

        Returns
        ----------
        Returns prediction confidences
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x = self.scaler.transform(x)

        confidences = self.model.predict_proba(x)

        confidences = torch.tensor(confidences, dtype=torch.float32)

        return confidences

    def inference(self, test_data_producer):
        """
        Perform model inference to get prediction confidences and true labels

        Parameters
        ----------
        @param test_data_producer: Data producer or data loader for test data

        Returns
        ----------
        Returns prediction confidences and true labels
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
        @param test_data_producer: Data producer or data loader for test data

        Returns
        ----------
        Returns prediction confidences and true labels
        """
        confidences = []
        gt_labels = []

        for x, y in test_data_producer:
            x = self.scaler.transform(x.numpy())
            confidence_batch = self.model.predict_proba(x)
            confidences.append(confidence_batch)
            gt_labels.append(y.numpy())

        return confidences, gt_labels

    def fit(self, train_data_producer):
        all_X_train = []
        all_y_train = []

        for batch_data in tqdm(train_data_producer, desc="Loading data"):
            X_batch, y_batch = batch_data
            all_X_train.append(X_batch.numpy())
            all_y_train.append(y_batch.numpy())

        X_train = np.vstack(all_X_train)
        y_train = np.hstack(all_y_train)

        print("X_train.shape:", X_train.shape)

        X_train = self.scaler.fit_transform(X_train)

        start_time = time.time()

        with tqdm(total=1, desc="Training RandomForest") as pbar:
            self.model.fit(X_train[:5000], y_train[:5000])
            pbar.update(1)

        end_time = time.time()

        training_time = end_time - start_time
        logger.info(
            f'{self.name} model trained successfully in {training_time} seconds.')

        self.save_model()

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

        x = x.detach().cpu().numpy()

        confidences = self.forward(x)

        return confidences, np.ones((confidences.size()[0],))

    def predict(self, test_data_producer, indicator_masking=True):
        """
        Predict labels and perform evaluation

        Parameters
        --------
        @param test_data_producer: torch.DataLoader, data loader for generating test data
        """
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1)

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

    def save_model(self):
        """
        Save the current model to disk.
        """
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

        with open(self.model_save_path, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)
        logger.info(f'Model saved to {self.model_save_path}')

    def load(self):
        """
        Load the model from disk.
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
