import os
import random
import tempfile

import numpy as np
import torch
from multiprocessing import Manager
from scipy.sparse.csr import csr_matrix
from sklearn.model_selection import train_test_split

from config import config
from core.droidfeature.feature_extraction import Apk2features
from tools import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, seed=0, device='cuda', feature_ext_args=None):
        """
        Construct dataset for machine learning model training.

        :param seed: Random seed
        :param device: Device type, 'cuda' or 'cpu'
        :param feature_ext_args: Parameters for feature extraction
        """

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        torch.set_default_dtype(torch.float32)

        self.temp_data = utils.SimplifyClass(Manager())

        self.device = device

        self.feature_ext_args = feature_ext_args
        if feature_ext_args is None:
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                  config.get('dataset', 'intermediate'))
        else:
            assert isinstance(feature_ext_args, dict)
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                  config.get(
                                                      'dataset', 'intermediate'),
                                                  **feature_ext_args)

        data_saving_path = os.path.join(config.get(
            'dataset', 'intermediate'), 'dataset.idx')

        if os.path.exists(data_saving_path) and (not self.feature_extractor.update):
            (self.train_dataset, self.validation_dataset,
             self.test_dataset) = utils.read_pickle(data_saving_path)

            def path_tran(data_paths):
                return np.array(
                    [os.path.join(config.get('metadata', 'naive_data_pool'),
                                  os.path.splitext(os.path.basename(name))[0] + self.feature_extractor.file_ext) for
                     name in data_paths])

            self.train_dataset = (
                path_tran(self.train_dataset[0]), self.train_dataset[1])
            self.validation_dataset = (
                path_tran(self.validation_dataset[0]), self.validation_dataset[1])
            self.test_dataset = (
                path_tran(self.test_dataset[0]), self.test_dataset[1])
        else:
            mal_feature_paths = self.apk_preprocess(
                config.get('dataset', 'malware_dir'))
            ben_feature_paths = self.apk_preprocess(
                config.get('dataset', 'benware_dir'))
            feature_paths = mal_feature_paths + ben_feature_paths

            gt_labels = np.zeros(
                (len(mal_feature_paths) + len(ben_feature_paths)), dtype=np.int32)
            gt_labels[:len(mal_feature_paths)] = 1

            self.train_dataset, self.validation_dataset, self.test_dataset = self.data_split(
                feature_paths, gt_labels)

            utils.dump_pickle(
                (self.train_dataset, self.validation_dataset, self.test_dataset), data_saving_path)

        self.vocab, _1, _2 = self.feature_extractor.get_vocab(
            *self.train_dataset)
        self.vocab_size = len(self.vocab)

        self.non_api_size = self.feature_extractor.get_non_api_size(self.vocab)

        self.n_classes = np.unique(self.train_dataset[1]).size

    def data_split(self, feature_paths, labels):
        """
        Split data into training, validation and test sets.

        :param feature_paths: List of feature file paths.
        :param labels: Corresponding list of labels.
        :return: (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
        """

        assert len(feature_paths) == len(labels)

        train_dn, validation_dn, test_dn = None, None, None

        data_split_path = os.path.join(config.get(
            'dataset', 'dataset_dir'), 'tr_te_va_split.name')

        if os.path.exists(data_split_path):
            train_dn, val_dn, test_dn = utils.read_pickle(data_split_path)

        if (train_dn is None) or (validation_dn is None) or (test_dn is None):
            data_names = [os.path.splitext(os.path.basename(path))[
                0] for path in feature_paths]

            train_dn, test_dn = train_test_split(
                data_names, test_size=0.2, random_state=self.seed, shuffle=True)

            train_dn, validation_dn = train_test_split(
                train_dn, test_size=0.25, random_state=self.seed, shuffle=True)

            utils.dump_pickle(
                (train_dn, validation_dn, test_dn), path=data_split_path)

        def query_path(_data_names):
            return np.array(
                [path for path in feature_paths if os.path.splitext(os.path.basename(path))[0] in _data_names])

        def query_indicator(_data_names):
            return [True if os.path.splitext(os.path.basename(path))[0] in _data_names else False for path in
                    feature_paths]

        train_data = query_path(train_dn)
        val_data = query_path(validation_dn)
        test_data = query_path(test_dn)

        random.seed(self.seed)
        random.shuffle(train_data)
        train_y = labels[query_indicator(train_dn)]
        random.seed(self.seed)
        random.shuffle(train_y)

        val_y = labels[query_indicator(validation_dn)]
        test_y = labels[query_indicator(test_dn)]

        return (train_data, train_y), (val_data, val_y), (test_data, test_y)

    def apk_preprocess(self, apk_paths, labels=None, update_feature_extraction=False):
        """
        Preprocess APK files.

        :param apk_paths: List of APK file paths.
        :param labels: List of corresponding labels, can be None.
        :param update_feature_extraction: Whether to update feature extractor status.
        :return: Processed feature paths, and optional labels.
        """

        old_status = self.feature_extractor.update

        self.feature_extractor.update = update_feature_extraction

        if labels is None:
            feature_paths = self.feature_extractor.feature_extraction(
                apk_paths)

            self.feature_extractor.update = old_status

            return feature_paths
        else:
            assert len(apk_paths) == len(labels), \
                'Mismatched data shape {} vs. {}'.format(len(apk_paths), len(labels))

            feature_paths = self.feature_extractor.feature_extraction(
                apk_paths)

            labels_ = []
            for i, feature_path in enumerate(feature_paths):
                fname = os.path.splitext(os.path.basename(feature_path))[0]

                if fname in apk_paths[i]:
                    labels_.append(labels[i])

            self.feature_extractor.update = old_status

            return feature_paths, np.array(labels_)

    def feature_preprocess(self, feature_paths):
        raise NotImplementedError

    def feature_api_rpst_sum(self, api_feat_representation_list):
        """
        Sum up API representations
        :param api_feat_representation_list: A list of sparse matrices
        """

        assert isinstance(api_feat_representation_list, list), "Expected input to be a list."

        if len(api_feat_representation_list) > 0:
            assert isinstance(api_feat_representation_list[0], csr_matrix)
        else:
            return np.zeros(shape=(self.vocab_size - self.non_api_size, self.vocab_size - self.non_api_size),
                            dtype=np.float)

        adj_array = np.asarray(
            api_feat_representation_list[0].todense()).astype(np.float32)

        for sparse_mat in api_feat_representation_list[1:]:
            adj_array += np.asarray(sparse_mat.todense()).astype(np.float32)

        return np.clip(adj_array, a_min=0, a_max=1)

    def get_numerical_input(self, feature_path, label):
        """
        loading features for given a feature path
        # results:
        # --->> mapping feature path to numerical representations
        # --->> features: 1d array, and a list of sparse matrices
        # --->> label: scalar
        """
        feature_vector, label = self.feature_extractor.feature2ipt(feature_path, label,
                                                                   self.vocab,
                                                                   None)
        return feature_vector, label

    def get_input_producer(self, feature_paths, y, batch_size, name='train', use_cache=False):
        """
        Get input producer, return a DataLoader object.

        :param feature_paths: List of feature paths.
        :param y: Labels.
        :param batch_size: Number of data in each batch.
        :param name: Usage scenario name, default is 'train'.
        :param use_cache: Whether to use cache, default is False.
        :return: Returns a DataLoader object.
        """

        params = {
            'batch_size': batch_size,
            'num_workers': self.feature_ext_args['proc_number'],
            'shuffle': False
        }

        use_cache = use_cache if name == 'train' else False

        return torch.utils.data.DataLoader(
            DatasetTorch(feature_paths, y, self,
                         name=name, use_cache=use_cache),
            worker_init_fn=lambda x: np.random.seed(
                torch.randint(0, 2**31, [1,])[0] + x),
            **params
        )

    def clear_up(self):
        self.temp_data.reset()

    @staticmethod
    def get_modification(adv_x, x, idx, sp=True):
        assert isinstance(adv_x, (np.ndarray, torch.Tensor))
        assert isinstance(x, (np.ndarray, torch.Tensor))

        x_mod = adv_x - x

        if isinstance(x_mod, np.ndarray):
            x_mod = np.array([x_mod[i, idx[i]] for i in range(x.shape[0])])
        else:
            x_mod = torch.stack([x_mod[i, idx[i]] for i in range(x.shape[0])])

        if sp:
            if isinstance(x_mod, torch.Tensor):
                return x_mod.to_sparse().cpu().unbind(dim=0)
            else:
                return torch.tensor(x_mod, dtype=torch.int).to_sparse().cpu().unbind(dim=0)
        else:
            if isinstance(x_mod, torch.Tensor):
                return x_mod.cpu().unbind(dim=0)
            else:
                return np.split(x_mod, x_mod.shape[0], axis=0)

    @staticmethod
    def modification_integ(x_mod_integrated, x_mod):
        assert isinstance(x_mod_integrated, list) and isinstance(x_mod, list)

        if len(x_mod_integrated) == 0:
            return x_mod

        assert len(x_mod_integrated) == len(x_mod)

        for i in range(len(x_mod)):
            assert not x_mod[i].is_cuda

            x_mod_integrated[i] += x_mod[i]

        return x_mod_integrated


class DatasetTorch(torch.utils.data.Dataset):
    '''Dataset defined for PyTorch'''

    def __init__(self, feature_paths, datay, dataset_obj, name='train', use_cache=False):
        '''Initialization method'''
        try:
            assert (name == 'train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only 'train', 'val', or 'test' are supported.\n")

        self.feature_paths = feature_paths
        self.datay = datay
        self.dataset_obj = dataset_obj
        self.name = name
        self.use_cache = use_cache

    def __len__(self):
        '''Return the total number of samples in the dataset'''
        return len(self.feature_paths)

    def __getitem__(self, index):
        '''Generate one data sample based on the index'''
        return self.get_item(index)

    def get_item(self, index):
        '''Get a data item'''
        if not self.use_cache:
            return self.dataset_obj.get_numerical_input(self.feature_paths[index], self.datay[index])
        elif self.dataset_obj.temp_data.is_cached(index):
            return self.dataset_obj.temp_data.get(index)
        else:
            feature_vec, label = \
                self.dataset_obj.get_numerical_input(
                    self.feature_paths[index], self.datay[index])
            self.dataset_obj.temp_data.cache(index, feature_vec, label)
            return feature_vec, label

    def reset_memory(self):
        '''Reset internal cache'''
        self.dataset_obj.temp_data.reset()
