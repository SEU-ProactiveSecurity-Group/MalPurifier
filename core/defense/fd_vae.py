# Import future features to ensure consistent behavior in Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import basic libraries
import random
import os.path as path

# Import PyTorch related libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data
from core.defense.amd_template import DetectorTemplate

# Import Captum library for model interpretability
from captum.attr import IntegratedGradients

# Import NumPy
import numpy as np

# Import configuration, logging and error handling from config module
from config import config, logging, ErrorHandler

# Import custom utility module
from tools import utils

from core.defense import Dataset

# Initialize logger and set its name
logger = logging.getLogger('core.defense.fd_vae')

# Add an error handler to the logger
logger.addHandler(ErrorHandler)


def get_label(length, IsBenign):
    '''
    Get labels.
    length: Number of labels.
    IsBenign: Whether the required label is benign.
    '''
    # Ensure length is an integer
    assert length-int(length) < 0.01
    length = int(length)

    # Generate vectors of ones and zeros of length 'length'
    ones = np.ones((length, 1))
    zeros = np.zeros((length, 1))

    # Return corresponding labels based on IsBenign
    if IsBenign:
        return np.column_stack((ones, zeros))  # Return benign labels if IsBenign is True
    else:
        return np.column_stack((zeros, ones))  # Return malicious labels if IsBenign is False


def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Variable: {name}, requires_grad: {param.requires_grad}")


class mu_sigma_MLP(nn.Module):
    # Initialization function
    def __init__(self,
                 num_epoch=30,
                 learn_rate=1e-3,
                 z_dim=20,
                 name='mlp'
                 ):
        super(mu_sigma_MLP, self).__init__()  # Call the initialization function of the parent class (nn.Module)

        self.num_epoch = num_epoch      # Set the number of training epochs
        self.batch_size = 128           # Set the size of each batch of data
        self.learn_rate = learn_rate    # Set the learning rate
        self.z_dim = z_dim              # Set the dimension of input data
        self.mu_wgt = 0.5               # Set the weight of mu
        self.sgm_wgt = 0.5              # Set the weight of sigma
        self.name = name                # Model name

        # Define neural network structure:
        # mu network for processing mu input
        self.mu_net = self._build_net(n_hidden=1000)

        # sigma network for processing sigma input
        self.sigma_net = self._build_net(n_hidden=1000)

        # Print model structure
        print('================================mu_sigma_MLP model architecture==============================')
        print(self)
        print('===============================================end==========================================')

        self.model_save_path = path.join(config.get(
            'experiments', 'mlp') + '_' + self.name, 'model.pth')

    # Define the sub-structure of the neural network, which includes 4 linear layers and intermediate activation functions
    def _build_net(self, n_hidden=128):
        # Use Sequential to construct a series of neural network modules
        return nn.Sequential(
            nn.Linear(self.z_dim, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Linear(n_hidden, 2)
        )

    # Forward propagation function
    def forward(self, mu_in, sigma_in):
        # Calculate mu output through mu network
        y1 = self.mu_net(mu_in)

        # Calculate sigma output through sigma network
        y2 = self.sigma_net(sigma_in)

        # Combine mu and sigma outputs, using weights for weighted average
        y = self.mu_wgt * y1 + self.sgm_wgt * y2

        return y

    def train_model(self, vae_model, train_data_producer, batch_size=128, n_epochs=50, verbose=True, device='cuda'):
        optimizer = optim.Adam(self.parameters())

        criterion_classification = nn.BCEWithLogitsLoss()

        # Reconstruction loss for VAE
        criterion_reconstruction = nn.MSELoss()
        best_accuracy = .0
        for epoch in range(n_epochs):
            for idx, (inputs, labels) in enumerate(train_data_producer):
                inputs, labels = inputs.to(device), labels.to(device)

                # Ensure inputs require gradients
                inputs.requires_grad_(True)

                optimizer.zero_grad()

                # Forward propagation
                y, muvae, sigmavae = vae_model.f(inputs)
                outputs = self(muvae, sigmavae)

                loss_reconstruction = criterion_reconstruction(y, inputs)

                # Convert labels to one-hot encoding
                labels = labels.long()
                one_hot_labels = torch.zeros(
                    labels.size(0), outputs.size(1)).to(device)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

                loss_classification = criterion_classification(
                    outputs, one_hot_labels)

                # Total loss
                loss = loss_reconstruction + loss_classification

                # Check if loss requires gradient
                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss tensor does not require gradients.")

                loss.backward()
                optimizer.step()

                if idx % 500 == 0:
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    print(predicted)
                    correct = (predicted == labels).sum().item()
                    accuracy = correct / len(labels)

                    # Printing
                    logger.info(
                        f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy

                        # Check if model save path exists, if not, create it
                        if not path.exists(self.model_save_path):
                            utils.mkdir(path.dirname(self.model_save_path))

                        # Save current model parameters
                        torch.save(self.state_dict(), self.model_save_path)

                        # If verbose mode is enabled, display model save path
                        if verbose:
                            print(f'Model saved at path: {self.model_save_path}')

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))


class VAE_2(nn.Module):
    def __init__(self,
                 dim_img=10000,
                 n_hidden=200,
                 dim_z=20,
                 KLW=5,
                 NLOSSW=10,
                 loss_type='1',
                 learn_rate=1e-3,
                 name='VAE_2'):
        super(VAE_2, self).__init__()

        # Initialize variables
        self.dim_img = dim_img          # Image dimension
        self.n_hidden = n_hidden        # Number of neurons in hidden layer
        self.dim_z = dim_z              # Dimension of latent space
        self.KLW = KLW                  # Weight of KL divergence
        self.NLOSSW = NLOSSW            # Weight of new loss function
        self.loss_type = loss_type      # Type of loss function
        self.learn_rate = learn_rate    # Learning rate
        self.name = name                # Model name
        self.mu = -1
        self.sigma = -1

        # Gaussian MLP encoder network structure
        self.fc1 = nn.Linear(self.dim_img, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc2_mu = nn.Linear(n_hidden, self.dim_z)
        self.fc2_sigma = nn.Linear(n_hidden, self.dim_z)

        # Bernoulli MLP decoder network structure
        self.fc3 = nn.Linear(self.dim_z, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, self.dim_img)

        # Print model structure
        print('========================================VAE model architecture==============================')
        print(self)
        print('===============================================end==========================================')

        # Define model save path
        self.model_save_path = path.join(config.get(
            'experiments', 'vae') + '_' + self.name, 'model.pth')

    def gaussian_MLP_encoder(self, x, keep_prob):
        h0 = F.elu(self.fc1(x))
        h0 = F.dropout(h0, p=1-keep_prob)

        h1 = F.tanh(self.fc2(h0))
        h1 = F.dropout(h1, p=1-keep_prob)

        # Calculate mean mu and variance sigma
        mu = self.fc2_mu(h1)
        sigma = 1e-6 + F.softplus(self.fc2_sigma(h1))

        return mu, sigma

    def bernoulli_MLP_decoder(self, z, keep_prob):
        h0 = F.tanh(self.fc3(z))
        h0 = F.dropout(h0, p=1-keep_prob)

        h1 = F.elu(self.fc4(h0))
        h1 = F.dropout(h1, p=1-keep_prob)

        # Output layer uses Sigmoid activation function to get output y in range [0,1]
        y = torch.sigmoid(self.fc5(h1))
        return y

    def forward(self, x_hat, x, x_comp, label_x, label_x_comp, keep_prob):
        # Encoder part:
        mu, sigma = self.gaussian_MLP_encoder(x_hat, keep_prob)
        mu1, sigma1 = self.gaussian_MLP_encoder(x_comp, keep_prob)
        muvae, sigmavae = self.gaussian_MLP_encoder(x, keep_prob)

        # Update mu and sigma values
        self.mu = muvae
        self.sigma = sigmavae

        # Sample z:
        # Use reparameterization trick of Gaussian distribution to sample z
        z = muvae + sigmavae * torch.randn_like(muvae)

        # Decoder part:
        y = self.bernoulli_MLP_decoder(z, keep_prob)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        # Calculate loss function:
        marginal_likelihood = torch.sum(
            x * torch.log(y) + (1 - x) * torch.log(1 - y), 1)
        KL_divergence = 0.5 * \
            torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2) - 1, 1)
        vector_loss = torch.mean((label_x_comp - label_x)**2, 1)
        loss_bac = 60 * vector_loss
        loss_mean = torch.mean((mu - mu1)**2, 1)

        # Calculate total loss based on vector loss and mean loss
        loss_0 = torch.mean(loss_mean * (1 - vector_loss))
        loss_1 = torch.mean(
            torch.abs(F.relu(loss_bac - loss_mean)) * vector_loss)

        # Calculate ELBO (Evidence Lower BOund)
        ELBO = self.KLW * \
            torch.mean(marginal_likelihood) - torch.mean(KL_divergence)
        loss = -ELBO

        # Update original loss with new loss
        new_loss = (loss_1 + loss_0) * self.NLOSSW
        if self.loss_type[0] == '1':
            loss = loss + new_loss

        return y, z, loss, torch.mean(marginal_likelihood), torch.mean(KL_divergence)

    def f(self, x):
        muvae, sigmavae = self.gaussian_MLP_encoder(x, 0.9)

        # Update mu and sigma values
        self.mu = muvae
        self.sigma = sigmavae

        # Sample z:
        # Use reparameterization trick of Gaussian distribution to sample z
        z = muvae + sigmavae * torch.randn_like(muvae)

        # Decoder part:
        y = self.bernoulli_MLP_decoder(z, 0.9)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)

        return y, muvae, sigmavae

    def train_model(self, train_data_producer, batch_size=128, n_epochs=10, verbose=True, device='cuda'):
        # Initialize optimizer, using Adam optimizer here
        optimizer = optim.Adam(self.parameters())

        ben_data_list = []
        mal_data_list = []

        # Traverse the DataLoader
        for _, (x_train, y_train) in enumerate(train_data_producer):
            ben_data = x_train[y_train == 0].to(device)
            mal_data = x_train[y_train == 1].to(device)

            ben_data_list.append(ben_data)
            mal_data_list.append(mal_data)
            mal_data_list.append(mal_data)  # Stack twice

        # Concatenate all batches together
        ben_data_combined = torch.cat(ben_data_list, 0).to(device)
        mal_data_combined = torch.cat(mal_data_list, 0).to(device)

        # Get the smaller number of samples to ensure equal number of benign and malicious samples
        n_samples = min(len(ben_data_combined), len(mal_data_combined))
        total_batch = n_samples // batch_size

        # Start training loop
        for epoch in range(n_epochs):
            # Set random seed
            random.seed(epoch)

            # Randomly shuffle the data
            indices_ben = torch.randperm(len(ben_data_combined))[:n_samples]
            indices_mal = torch.randperm(len(mal_data_combined))[:n_samples]

            # Get shuffled data
            ben_data_combined = ben_data_combined[indices_ben]
            mal_data_combined = mal_data_combined[indices_mal]

            # Get batch labels
            lbBen = get_label(batch_size, True)
            lbMal = get_label(batch_size, False)
            batch_label = np.row_stack((lbBen, lbMal))

            for i in range(total_batch):
                offset = (i * batch_size) % n_samples

                # Get benign and malicious batch data
                batch_ben_input_s = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_mal_input_s = mal_data_combined[offset:(
                    offset + batch_size), :]
                batch_ben_input_s_cpu = batch_ben_input_s.cpu().numpy()
                batch_mal_input_s_cpu = batch_mal_input_s.cpu().numpy()
                batch_input = np.row_stack(
                    (batch_ben_input_s_cpu, batch_mal_input_s_cpu))

                # Combine input data and labels
                batch_input_wl = np.column_stack((batch_input, batch_label))
                np.random.shuffle(batch_input_wl)  # Shuffle combined data

                # Separate input data and labels
                batch_xs_input = batch_input_wl[:, :-2]
                batch_xs_label = batch_input_wl[:, -2:]

                # Get next batch data for comparison
                offset = ((i + 1) * batch_size) % (n_samples - batch_size)
                batch_ben_input_s = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_mal_input_s = mal_data_combined[offset:(
                    offset + batch_size), :]
                batch_ben_input_s_cpu = batch_ben_input_s.cpu().numpy()
                batch_mal_input_s_cpu = batch_mal_input_s.cpu().numpy()
                batch_input = np.row_stack(
                    (batch_ben_input_s_cpu, batch_mal_input_s_cpu))

                batch_input_wl = np.column_stack((batch_input, batch_label))
                np.random.shuffle(batch_input_wl)  # Shuffle combined data

                # Separate input data and labels
                batch_xcomp_input = batch_input_wl[:, :-2]
                batch_xcomp_label = batch_input_wl[:, -2:]

                # Get target data
                batch_xs_target = ben_data_combined[offset:(
                    offset + batch_size), :]
                batch_xs_target = torch.cat(
                    (batch_xs_target, batch_xs_target), dim=0)
                assert batch_xs_input.shape == batch_xs_target.shape

                # Zero out previous gradients
                optimizer.zero_grad()

                if isinstance(batch_xs_input, np.ndarray):
                    batch_xs_input = torch.tensor(
                        batch_xs_input, dtype=torch.float32).to(device)

                if isinstance(batch_xs_target, np.ndarray):
                    batch_xs_target = torch.tensor(
                        batch_xs_target, dtype=torch.float32).to(device)

                if isinstance(batch_xs_label, np.ndarray):
                    batch_xs_label = torch.tensor(
                        batch_xs_label, dtype=torch.float32).to(device)

                if isinstance(batch_xcomp_label, np.ndarray):
                    batch_xcomp_label = torch.tensor(
                        batch_xcomp_label, dtype=torch.float32).to(device)

                if isinstance(batch_xcomp_input, np.ndarray):
                    batch_xcomp_input = torch.tensor(
                        batch_xcomp_input, dtype=torch.float32).to(device)

                self.x_hat = batch_xs_input
                self.x = batch_xs_target
                self.x_comp = batch_xcomp_input
                self.label_x = batch_xs_label
                self.label_x_comp = batch_xcomp_label
                self.keep_prob = 0.9

                # Forward propagation
                y, z, loss, marginal_likelihood, KL_divergence = self.forward(batch_xs_input,
                                                                              batch_xs_target,
                                                                              batch_xcomp_input,
                                                                              batch_xs_label,
                                                                              batch_xcomp_label,
                                                                              0.9)

                # Backward propagation
                loss.backward()

                # Update weights
                optimizer.step()

            # Print loss information for each epoch
            logger.info(
                f"epoch[{epoch}/{n_epochs}]: L_tot {loss.item():.2f} L_likelihood {marginal_likelihood.item():.2f} L_divergence {KL_divergence.item():.2f}")

        # Check if model save path exists, if not, create it
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

        # Save current model parameters
        torch.save(self.state_dict(), self.model_save_path)

        # If verbose mode is enabled, display model save path
        if verbose:
            print(f'Model saved at path: {self.model_save_path}')

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))


class VAE_SU(nn.Module, DetectorTemplate):
    def __init__(self,
                 input_size=10000,
                 n_hidden=200,
                 n_epochs=50,
                 z_dim=20,
                 learn_rate=1e-3,
                 Loss_type='1',
                 KLW=10,
                 NLOSSW=10,
                 name='fd_vae',
                 device='cuda:0',
                 **kwargs):

        super(VAE_SU, self).__init__()
        DetectorTemplate.__init__(self)

        # Remove kwargs as it seems unused
        del kwargs

        self.nb_classes = 2

        self.hparams = locals()
        self.device = device
        self.name = name
        self.tau = 500

        # Initialize and build the MLP model
        self.Mlp = mu_sigma_MLP(num_epoch=n_epochs,
                                learn_rate=learn_rate,
                                z_dim=z_dim,
                                name=name
                                )

        # Initialize and build the VAE model
        self.Vae = VAE_2(dim_img=input_size,
                         n_hidden=n_hidden,
                         dim_z=z_dim,
                         KLW=KLW,
                         NLOSSW=NLOSSW,
                         loss_type=Loss_type,
                         learn_rate=1e-3,
                         name=name
                         )

        # Define model save path
        self.model_save_path = path.join(config.get('experiments', 'fd_vae') + '_' + self.name,
                                         'model.pth')

        # Log model structure information
        logger.info(
            '=====================================fd_vae model architecture=============================')
        logger.info(self)
        logger.info(
            '===============================================end==========================================')

        self.dim = self.Vae.dim_img

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau  # Return tau, i.e., decision threshold

    def forward(self, x, a=None, **kwargs):
        self.Vae.load()
        self.Mlp.load()

        x = x.float()
        y, muvae, sigmavae = self.Vae.f(x)
        outputs = self.Mlp(muvae, sigmavae)
        loss_reconstruction = ((y - x) ** 2).sum(dim=-1)

        x_cent = torch.softmax(outputs, dim=-1)

        return x_cent, loss_reconstruction

    def fit(self, train_data_producer, verbose=True):
        # train Vae
        vae_model = self.Vae
        self.Vae.train_model(train_data_producer, device=self.device)

        # train Mlp
        self.Mlp.train_model(
            vae_model, train_data_producer, device=self.device)

        # Check if model save path exists, if not, create it
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        # Save current model parameters
        torch.save(self.state_dict(), self.model_save_path)
        # If verbose mode is enabled, display model save path
        if verbose:
            print(f'Model saved at path: {self.model_save_path}')

    def load(self):
        """
        Load model parameters from disk
        """
        self.Vae.load_state_dict(torch.load(self.model_save_path))
        self.Mlp.load_state_dict(torch.load(self.model_save_path))

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []  # Store true labels for each batch of data

        self.Vae.load()
        self.Mlp.load()

        with torch.no_grad():
            for x, l in test_data_producer:
                x, l = utils.to_device(x, l, self.device)

                x_cent, loss_reconstruction = self.forward(x)
                y_cent.append(x_cent)
                x_prob.append(loss_reconstruction)
                # Store the actual labels instead of the VAE output
                gt_labels.append(l)

        # Stack confidence values from all batches into a tensor
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)

        # Concatenate true labels from all batches into a tensor
        gt_labels = torch.cat(gt_labels, dim=0)

        return y_cent, x_prob, gt_labels

    # Only supports batch inference for malware samples

    def inference_batch_wise(self, x):
        assert isinstance(x, torch.Tensor)
        x = x.float()
        self.Vae.load()
        self.Mlp.load()

        y, muvae, sigmavae = self.Vae.f(x)
        outputs = self.Mlp(muvae, sigmavae)

        x_cent_values = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
        x_cent = x_cent_values[:2] if len(
            x_cent_values.shape) == 1 else x_cent_values[:, :2]

        loss_reconstruction = ((y - x) ** 2).sum(dim=-1)
        reloss = loss_reconstruction.detach().cpu().numpy()

        return x_cent, reloss

    def indicator(self, reloss, y_pred=None):
        # Manually set metric
        metric = 500
        if isinstance(reloss, np.ndarray):
            reloss = torch.tensor(reloss, device=self.device)
            metric_tensor = torch.tensor(metric).to(reloss.device)
            return (reloss <= metric_tensor).cpu().numpy()
        elif isinstance(reloss, torch.Tensor):
            return reloss <= metric
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    # Predict labels and perform evaluation

    def predict(self, test_data_producer, indicator_masking=True, metric=5000):
        y_cent, reloss, y_true = self.inference(test_data_producer)

        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        # Get indicator flags for subsequent masking or filtering
        indicator_flag = self.indicator(reloss, metric).cpu().numpy()

        # Define an internal function to evaluate model performance
        def measurement(_y_true, _y_pred):
            # Import required evaluation metric libraries
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score

            # Calculate and print accuracy and balanced accuracy
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            logger.info(
                f"The accuracy on the test dataset is {accuracy * 100:.5f}%")
            logger.info(
                f"The balanced accuracy on the test dataset is {b_accuracy * 100:.5f}%")

            # Check if all classes are present in true labels
            if np.any([np.all(_y_true == i) for i in range(self.nb_classes)]):
                logger.warning("class absent.")
                return

            # Calculate confusion matrix and get various metrics from it
            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(_y_true, _y_pred, average='binary')

            # Print other potentially needed evaluation metrics
            logger.info(f"False Negative Rate (FNR) is {fnr * 100:.5f}%, \
                        False Positive Rate (FPR) is {fpr * 100:.5f}%, F1 score is {f1 * 100:.5f}%")

        # Perform evaluation for the first time
        measurement(y_true, y_pred)

        # Decide how to handle the indicator based on indicator_masking
        if indicator_masking:
            # Exclude examples with "uncertain" responses
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # Here, instead of filtering out samples, reset their predictions to 1
            y_pred[~indicator_flag] = 1.

        # Print indicator status and threshold information
        logger.info('The indicator is turning on...')

        # Perform evaluation again
        measurement(y_true, y_pred)

    def load(self):
        self.load_state_dict(torch.load(self.model_save_path))
