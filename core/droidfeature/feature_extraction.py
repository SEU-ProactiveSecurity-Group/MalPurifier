import os.path
from tqdm import tqdm
import multiprocessing

import sys
import os

import collections
import numpy as np

from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.droidfeature import feature_gen
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.droidfeature.feature_extraction')
logger.addHandler(ErrorHandler)


class Apk2features(object):
    """Get features from an APK"""

    def __init__(self,
                 naive_data_save_dir, # 用于保存中间数据的目录
                 intermediate_save_dir, # 用于保存特征 pickle 文件的目录
                 number_of_smali_files=1000000, # 处理的 smali 文件的最大数量，默认为 1000000。
                 max_vocab_size=10000, # 词汇表的最大大小，默认为 10000
                 file_ext='.feat', # 文件扩展名，默认为 '.feat'
                 update=False,  # 表示是否重新计算原始特征，默认为 False
                 proc_number=2, # 进程数，默认为 2
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

    # 这段代码定义了 Apk2features 类的 feature_extraction 方法，
    # 用于从指定目录中的 APK 文件中提取特征并保存。方法返回提取特征后的文件路径。
    def feature_extraction(self, sample_dir):
        """ save the android features and return the saved paths """
        sample_path_list = utils.check_dir(sample_dir)
        pool = multiprocessing.Pool(self.proc_number, initializer=utils.pool_initializer)

        # 定义一个名为 get_save_path 的内部函数，用于获取特征保存路径。
        # 它根据 APK 文件的 SHA256 编码和文件扩展名生成保存路径。
        # 如果该路径对应的文件已存在，并且不需要更新特征，则返回 None。否则，返回保存路径。
        def get_save_path(a_path):
            sha256_code = os.path.splitext(os.path.basename(a_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)

            if os.path.exists(save_path) and (not self.update):
                return
            else:
                return save_path
            
        # 创建一个名为 params 的列表，包含需要提取特征的 APK 文件路径、处理的 smali 文件最大数量和特征保存路径。
        # 只有当 get_save_path 返回值不为 None 时，才将 APK 文件路径添加到 params 列表中。
        params = [(apk_path, self.number_of_smali_files, get_save_path(apk_path)) for \
                  apk_path in sample_path_list if get_save_path(apk_path) is not None]
        
        # 使用 pool.imap_unordered() 方法并行地对 params 中的每个元素执行 feature_gen.apk2feat_wrapper 函数。
        # 使用 tqdm 显示处理进度。如果处理过程中出现异常，使用 logger.error 输出错误信息。
        for res in tqdm(pool.imap_unordered(feature_gen.apk2feat_wrapper, params), total=len(params)):
            if isinstance(res, Exception):
                logger.error("Failed processing: {}".format(str(res)))
        pool.close()
        pool.join()

        feature_paths = []
        
        # 遍历 sample_path_list，获取每个 APK 文件的特征保存路径。
        # 如果路径对应的文件存在，则将其添加到 feature_paths 列表中。
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
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
        
        feature_path_list：特征文件路径列表，每个路径指向一个特征文件。
        gt_labels：真实标签，表示每个特征文件对应的恶意软件或良性样本。
        方法返回一个包含词汇表、词汇信息和词汇类型的元组。
        """
        vocab_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab')
        vocab_type_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab_type')
        vocab_extra_info_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab_info')
        
        # 如果这些文件已经存在且不需要更新，从文件中读取并返回词汇表、词汇信息和词汇类型。
        if os.path.exists(vocab_saving_path) and os.path.exists(vocab_saving_path) and (not self.update):
            return utils.read_pickle(vocab_saving_path), utils.read_pickle(vocab_extra_info_saving_path), utils.read_pickle(vocab_type_saving_path)
        elif feature_path_list is None and gt_labels is None:
            raise FileNotFoundError("No vocabulary found and no features for producing vocabulary!")
        else:
            pass
        
        # 确保输入的恶意软件和良性样本标签都存在，并检查
        # feature_path_list 和 gt_labels 的长度是否相等。
        assert not (np.all(gt_labels == 1) or np.all(gt_labels == 0)), 'Expect both malware and benign samples.'
        assert len(feature_path_list) == len(gt_labels)

        # 使用 collections.Counter 和 collections.defaultdict 创建计数器和字典以存储词汇表相关信息。
        counter_mal, counter_ben = collections.Counter(), collections.Counter()
        feat_info_dict = collections.defaultdict(set)
        feat_type_dict = collections.defaultdict(str)
        
        # 遍历 feature_path_list 和 gt_labels
        for feature_path, label in zip(feature_path_list, gt_labels):
            if not os.path.exists(feature_path):
                continue
            features = feature_gen.read_from_disk(feature_path)
            # 获取特征列表、特征信息列表和特征类型列表。
            # 根据标签更新恶意软件和良性样本的计数器。
            feature_list, feature_info_list, feature_type_list = feature_gen.get_feature_list(features)
            feature_occurrence = list(dict.fromkeys(feature_list))
            for _feat, _feat_info, _feat_type in zip(feature_list, feature_info_list, feature_type_list):
                feat_info_dict[_feat].add(_feat_info)
                feat_type_dict[_feat] = _feat_type
            if label:
                counter_mal.update(list(feature_occurrence))
            else:
                counter_ben.update(list(feature_occurrence))
        all_words = list(dict.fromkeys(list(counter_ben.keys()) + list(counter_mal.keys())))
        if len(all_words) <= 0:
            raise ValueError("No features exist on this dataset.")

        # 根据特征选择策略选择词汇
        maximum_vocab_size = self.maximum_vocab_size
        selected_words = []
        
        # ----------------------------------------
        # dangerous permission
        # 危险权限选择：提取词汇表中的危险权限特征，并对每个权限进行检查。
        # 如果权限被认为是危险的（通过 feature_gen.permission_check 函数判断），
        # 则将其添加到 selected_words 列表中。
        all_words_type = list(map(feat_type_dict.get, all_words))
        perm_pos = np.array(all_words_type)[...] == feature_gen.PERMISSION
        perm_features = np.array(all_words)[perm_pos]
        for perm in perm_features:
            if feature_gen.permission_check(perm):
                selected_words.append(perm)

        # intent
        # 意图选择：提取词汇表中的意图特征，并对每个意图进行检查。
        # 如果意图被认为是有害的（通过 feature_gen.intent_action_check 函数判断），
        # 则将其添加到 selected_words 列表中。
        intent_pos = np.array(all_words_type)[...] == feature_gen.INTENT
        intent_features = np.array(all_words)[intent_pos]
        for intent in intent_features:
            if feature_gen.intent_action_check(intent):
                selected_words.append(intent)

        # suspicious apis
        # 可疑 API 选择：提取词汇表中的系统 API 特征，并对每个 API 进行检查。
        # 如果 API 被认为是可疑的或敏感的（通过 feature_gen.check_suspicious_api 或 feature_gen.check_sensitive_api 函数判断），
        # 则将其添加到 selected_words 列表中。
        api_pos = np.array(all_words_type)[...] == feature_gen.SYS_API
        susp_apis = np.array(all_words)[api_pos]
        for api in susp_apis:
            if feature_gen.check_suspicious_api(api) or feature_gen.check_sensitive_api(api):
                selected_words.append(api)
        # ----------------------------------------
        
        # remove components
        # 移除组件：从词汇表中移除所有属于活动、服务、接收器和提供器的组件。
        api_comps = np.array(all_words_type)[...] == feature_gen.ACTIVITY
        api_comps = api_comps | (np.array(all_words_type)[...] == feature_gen.SERVICE)
        api_comps = api_comps | (np.array(all_words_type)[...] == feature_gen.RECEIVER)
        api_comps = api_comps | (np.array(all_words_type)[...] == feature_gen.PROVIDER)
                
        # 计算恶意软件和良性样本的特征频率差并根据差异对词汇进行排序。
        # 选择最多 maximum_vocab_size 个词汇。
        all_words = list(np.array(all_words)[~api_comps])
        for s_word in selected_words:
            all_words.remove(s_word)
        logger.info("The total number of words: {}-{}.".format(len(selected_words), len(all_words)))

        # 计算恶意样本的特征频率
        mal_feature_frequency = np.array(list(map(counter_mal.get, all_words)))
        mal_feature_frequency[mal_feature_frequency == None] = 0
        mal_feature_frequency = mal_feature_frequency.astype(np.float64)
        mal_feature_frequency /= np.sum(gt_labels)

        # 计算良性样本的特征频率
        ben_feature_frequency = np.array(list(map(counter_ben.get, all_words)))
        ben_feature_frequency[ben_feature_frequency == None] = 0
        ben_feature_frequency = ben_feature_frequency.astype(np.float64)
        ben_feature_frequency /= float(len(gt_labels) - np.sum(gt_labels))

        # 计算特征频率差
        feature_freq_diff = abs(mal_feature_frequency - ben_feature_frequency)

        # 根据特征频率差进行排序
        posi_selected = np.argsort(feature_freq_diff)[::-1]
        ordered_words = selected_words + [all_words[p] for p in posi_selected]

        # 选择最多 maximum_vocab_size 个词汇
        selected_words = ordered_words[:maximum_vocab_size]

        # 获取所选词汇的类型和对应的词汇信息：
        # 使用 feat_type_dict 和 feat_info_dict 字典分别获取所选词汇的类型和对应的词汇信息，以便在之后的处理中使用。
        selected_word_type = list(map(feat_type_dict.get, selected_words))
        corresponding_word_info = list(map(feat_info_dict.get, selected_words))

        # 保存所选词汇、词汇类型和对应词汇信息到文件，然后返回这些值
        if len(selected_words) > 0:
            utils.dump_pickle(selected_words, vocab_saving_path)
            utils.dump_pickle(selected_word_type, vocab_type_saving_path)
            utils.dump_pickle(corresponding_word_info, vocab_extra_info_saving_path)
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

    # ⭐ 这段代码定义了一个名为 feature2ipt 的方法，它将应用程序的特征映射到数值表示。
    # feature2ipt 方法的主要目的是将应用程序的特征映射到一个固定长度的向量，
    # 其中每个元素表示对应词汇表中单词的存在（1）或不存在（0）。
    # 这样的数值表示可以作为机器学习模型的输入，以便对应用程序进行分类或其他分析任务。
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
        # 确保词汇表不为空
        assert vocabulary is not None and len(vocabulary) > 0
        
        # 检查缓存目录是否存在，如果存在则加载缓存数据
        if isinstance(cache_dir, str):
            rpst_cached_name = self.get_cached_name(feature_path)
            rpst_cached_path = os.path.join(cache_dir, rpst_cached_name)
            if os.path.exists(rpst_cached_path):
                return utils.read_pickle(rpst_cached_path, use_gzip=True)
            
        # 如果 feature_path 无效，则返回零向量表示
        if not isinstance(feature_path, str):
            logger.warning("Cannot find the feature path: {}, zero vector used".format(feature_path))
            return np.zeros((len(vocabulary), ), dtype=np.float32), []

        if not os.path.exists(feature_path):
            logger.warning("Cannot find the feature path: {}, zero vector used".format(feature_path))
            return np.zeros((len(vocabulary), ), dtype=np.float32), []

        # 从给定的 feature_path 加载原始特征，并将其格式化为非 API 特征和 API 特征。
        native_features = feature_gen.read_from_disk(feature_path)
        non_api_features, api_features = feature_gen.format_feature(native_features)
        features = non_api_features + api_features

        # 初始化一个长度与词汇表相等的零向量（representation_vector）作为数值表示。
        representation_vector = np.zeros((len(vocabulary), ), dtype=np.float32)
        
        # 将词汇表映射到其索引，并根据提取到的特征填充 representation_vector。
        dictionary = dict(zip(vocabulary, range(len(vocabulary))))
        filled_pos = [idx for idx in list(map(dictionary.get, features)) if idx is not None]
        
        if len(filled_pos) > 0:
            representation_vector[filled_pos] = 1.
        return representation_vector, label

def _main():
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # sys.path.insert(0, project_root)
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    
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
    
    # 获取词汇表 vocab
    # 🐖 参数对不上
    # vocab, _1 = feature_extractor.get_vocab(mal_paths + ben_paths, labels)
    vocab, vocab1, vocab2 = feature_extractor.get_vocab(mal_paths + ben_paths, labels)
    # pprint(vocab)
    # pprint(vocab1)
    # pprint(vocab2)
    
    # 使用 feature_extractor.feature2ipt() 方法将恶意软件目录中的第一个 APK 文件的特征转换为输入表示，
    # 同时传入词汇表 vocab。结果存储在 n_rpst 和 api_rpst 中。
    # 🐖 参数对不上
    # n_rpst, api_rpst, _1 = feature_extractor.feature2ipt(mal_paths[0], label=1, vocabulary=vocab)
    
    for i in range(len(mal_paths)):
        n_rpst, api_rpst = feature_extractor.feature2ipt(mal_paths[i], label=1, vocabulary=vocab)
        print(n_rpst)
        print(n_rpst.shape)
        print(api_rpst)
    
    # print(api_rpst)

if __name__ == "__main__":
    import sys

    sys.exit(_main())