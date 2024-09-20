from config import logging, ErrorHandler
from tools import utils
from core.droidfeature import feature_gen
import os.path
from tqdm import tqdm
import multiprocessing

import sys
import os

import collections
import numpy as np

from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


logger = logging.getLogger('core.droidfeature.feature_extraction')
logger.addHandler(ErrorHandler)


class Apk2features(object):
    """Get features from an APK"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 number_of_smali_files=1000000,
                 max_vocab_size=10000,
                 file_ext='.feat',
                 update=False,
                 proc_number=2,
                 **kwargs
                 ):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving feature pickle files
        :param number_of_smali_files: the maximum number of smali files processed
        :param max_vocab_size: the maximum number of words
        :param file_ext: file extension
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        self.naive_data_save_dir = naive_data_save_dir
        self.intermediate_save_dir = intermediate_save_dir
        self.maximum_vocab_size = max_vocab_size
        self.number_of_smali_files = number_of_smali_files

        self.file_ext = file_ext
        self.update = update
        self.proc_number = proc_number

        if len(kwargs) > 0:
            logger.warning("unused hyper parameters {}.".format(kwargs))

    def feature_extraction(self, sample_dir):
        """ save the android features and return the saved paths """
        sample_path_list = utils.check_dir(sample_dir)
        pool = multiprocessing.Pool(
            self.proc_number, initializer=utils.pool_initializer)

        def get_save_path(a_path):
            sha256_code = os.path.splitext(os.path.basename(a_path))[0]
            save_path = os.path.join(
                self.naive_data_save_dir, sha256_code + self.file_ext)

            if os.path.exists(save_path) and (not self.update):
                return
            else:
                return save_path

        params = [(apk_path, self.number_of_smali_files, get_save_path(apk_path)) for
                  apk_path in sample_path_list if get_save_path(apk_path) is not None]

        for res in tqdm(pool.imap_unordered(feature_gen.apk2feat_wrapper, params), total=len(params)):
            if isinstance(res, Exception):
                logger.error("Failed processing: {}".format(str(res)))
        pool.close()
        pool.join()

        feature_paths = []

        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]
            save_path = os.path.join(
                self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_paths.append(save_path)

        return feature_paths

    def get_vocab(self, feature_path_list=None, gt_labels=None):
        """
        get vocabularies incorporating feature selection
        :param feature_path_list: feature_path_list, list, a list of paths, 
        each of which directs to a feature file (we \
        suggest using the feature files for the training purpose)
        :param gt_labels: gt_labels, list or numpy.ndarray, ground truth labels
        :return: list, a list of words
        """
        vocab_saving_path = os.path.join(
            self.intermediate_save_dir, 'data.vocab')
        vocab_type_saving_path = os.path.join(
            self.intermediate_save_dir, 'data.vocab_type')
        vocab_extra_info_saving_path = os.path.join(
            self.intermediate_save_dir, 'data.vocab_info')

        if os.path.exists(vocab_saving_path) and os.path.exists(vocab_saving_path) and (not self.update):
            return utils.read_pickle(vocab_saving_path), utils.read_pickle(vocab_extra_info_saving_path), utils.read_pickle(vocab_type_saving_path)
        elif feature_path_list is None and gt_labels is None:
            raise FileNotFoundError(
                "No vocabulary found and no features for producing vocabulary!")
        else:
            pass

        assert not (np.all(gt_labels == 1) or np.all(gt_labels == 0)
                    ), 'Expect both malware and benign samples.'
        assert len(feature_path_list) == len(gt_labels)

        counter_mal, counter_ben = collections.Counter(), collections.Counter()
        feat_info_dict = collections.defaultdict(set)
        feat_type_dict = collections.defaultdict(str)

        for feature_path, label in zip(feature_path_list, gt_labels):
            if not os.path.exists(feature_path):
                continue
            features = feature_gen.read_from_disk(feature_path)
            feature_list, feature_info_list, feature_type_list = feature_gen.get_feature_list(
                features)
            feature_occurrence = list(dict.fromkeys(feature_list))
            for _feat, _feat_info, _feat_type in zip(feature_list, feature_info_list, feature_type_list):
                feat_info_dict[_feat].add(_feat_info)
                feat_type_dict[_feat] = _feat_type
            if label:
                counter_mal.update(list(feature_occurrence))
            else:
                counter_ben.update(list(feature_occurrence))
        all_words = list(dict.fromkeys(
            list(counter_ben.keys()) + list(counter_mal.keys())))
        if len(all_words) <= 0:
            raise ValueError("No features exist on this dataset.")

        maximum_vocab_size = self.maximum_vocab_size
        selected_words = []

        all_words_type = list(map(feat_type_dict.get, all_words))
        perm_pos = np.array(all_words_type)[...] == feature_gen.PERMISSION
        perm_features = np.array(all_words)[perm_pos]
        for perm in perm_features:
            if feature_gen.permission_check(perm):
                selected_words.append(perm)

        intent_pos = np.array(all_words_type)[...] == feature_gen.INTENT
        intent_features = np.array(all_words)[intent_pos]
        for intent in intent_features:
            if feature_gen.intent_action_check(intent):
                selected_words.append(intent)

        api_pos = np.array(all_words_type)[...] == feature_gen.SYS_API
        susp_apis = np.array(all_words)[api_pos]
        for api in susp_apis:
            if feature_gen.check_suspicious_api(api) or feature_gen.check_sensitive_api(api):
                selected_words.append(api)

        api_comps = np.array(all_words_type)[...] == feature_gen.ACTIVITY
        api_comps = api_comps | (
            np.array(all_words_type)[...] == feature_gen.SERVICE)
        api_comps = api_comps | (
            np.array(all_words_type)[...] == feature_gen.RECEIVER)
        api_comps = api_comps | (
            np.array(all_words_type)[...] == feature_gen.PROVIDER)

        all_words = list(np.array(all_words)[~api_comps])
        for s_word in selected_words:
            all_words.remove(s_word)
        logger.info(
            "The total number of words: {}-{}.".format(len(selected_words), len(all_words)))

        mal_feature_frequency = np.array(list(map(counter_mal.get, all_words)))
        mal_feature_frequency[mal_feature_frequency == None] = 0
        mal_feature_frequency = mal_feature_frequency.astype(np.float64)
        mal_feature_frequency /= np.sum(gt_labels)

        ben_feature_frequency = np.array(list(map(counter_ben.get, all_words)))
        ben_feature_frequency[ben_feature_frequency == None] = 0
        ben_feature_frequency = ben_feature_frequency.astype(np.float64)
        ben_feature_frequency /= float(len(gt_labels) - np.sum(gt_labels))

        feature_freq_diff = abs(mal_feature_frequency - ben_feature_frequency)

        posi_selected = np.argsort(feature_freq_diff)[::-1]
        ordered_words = selected_words + [all_words[p] for p in posi_selected]

        selected_words = ordered_words[:maximum_vocab_size]

        selected_word_type = list(map(feat_type_dict.get, selected_words))
        corresponding_word_info = list(map(feat_info_dict.get, selected_words))

        if len(selected_words) > 0:
            utils.dump_pickle(selected_words, vocab_saving_path)
            utils.dump_pickle(selected_word_type, vocab_type_saving_path)
            utils.dump_pickle(corresponding_word_info,
                              vocab_extra_info_saving_path)
        return selected_words, corresponding_word_info, selected_word_type

    def feature_mapping(self, feature_path_list, dictionary):
        """
        mapping feature to numerical representation
        :param feature_path_list: a list of feature paths
        :param dictionary: vocabulary -> index
        :return: 2D representation
        :rtype numpy.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def get_non_api_size(vocabulary=None):
        cursor = 0
        for word in vocabulary:
            if '->' not in word:  # exclude the api features
                cursor += 1
            else:
                break
        return cursor

    def get_cached_name(self, feature_path):
        if os.path.isfile(feature_path):
            return os.path.splitext(os.path.basename(feature_path))[0] + '.npz'
        else:
            raise FileNotFoundError

    def feature2ipt(self, feature_path, label, vocabulary=None, cache_dir=None):
        """
        Map features to numerical representations

        Parameters
        --------
        :param feature_path, string, a path directs to a feature file
        :param label, int, ground truth labels
        :param vocabulary:list, a list of words
        :param cache_dir: a temporal folder
        :return: numerical representations corresponds to an app. Each representation contains a tuple
        (feature 1D array, label)
        """
        assert vocabulary is not None and len(vocabulary) > 0

        if isinstance(cache_dir, str):
            rpst_cached_name = self.get_cached_name(feature_path)
            rpst_cached_path = os.path.join(cache_dir, rpst_cached_name)
            if os.path.exists(rpst_cached_path):
                return utils.read_pickle(rpst_cached_path, use_gzip=True)

        if not isinstance(feature_path, str):
            logger.warning(
                "Cannot find the feature path: {}, zero vector used".format(feature_path))
            return np.zeros((len(vocabulary), ), dtype=np.float32), []

        if not os.path.exists(feature_path):
            logger.warning(
                "Cannot find the feature path: {}, zero vector used".format(feature_path))
            return np.zeros((len(vocabulary), ), dtype=np.float32), []

        native_features = feature_gen.read_from_disk(feature_path)
        non_api_features, api_features = feature_gen.format_feature(
            native_features)
        features = non_api_features + api_features

        representation_vector = np.zeros((len(vocabulary), ), dtype=np.float32)

        dictionary = dict(zip(vocabulary, range(len(vocabulary))))
        filled_pos = [idx for idx in list(
            map(dictionary.get, features)) if idx is not None]

        if len(filled_pos) > 0:
            representation_vector[filled_pos] = 1.
        return representation_vector, label


def _main():
    from config import config

    malware_dir_name = config.get('dataset', 'malware_dir')
    benign_dir_name = config.get('dataset', 'benware_dir')
    meta_data_saving_dir = config.get('dataset', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')

    feature_extractor = Apk2features(naive_data_saving_dir,
                                     meta_data_saving_dir,
                                     update=False,
                                     proc_number=2)

    mal_paths = feature_extractor.feature_extraction(malware_dir_name)
    pprint(mal_paths)

    ben_paths = feature_extractor.feature_extraction(benign_dir_name)
    pprint(ben_paths)

    labels = np.zeros((len(mal_paths) + len(ben_paths), ))
    labels[:len(mal_paths)] = 1
    pprint(labels)

    vocab, vocab1, vocab2 = feature_extractor.get_vocab(
        mal_paths + ben_paths, labels)

    for i in range(len(mal_paths)):
        n_rpst, api_rpst = feature_extractor.feature2ipt(
            mal_paths[i], label=1, vocabulary=vocab)
        print(n_rpst)
        print(n_rpst.shape)
        print(api_rpst)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
