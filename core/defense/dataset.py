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
        为机器学习模型学习构建数据集。
        
        :param seed: 随机种子
        :param device: 设备类型，'cuda' 或 'cpu'
        :param feature_ext_args: 提取特征的参数
        """
        
        # 设置随机种子，并确保随机性在不同库之间是一致的
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 设置PyTorch的默认数据类型为float32
        torch.set_default_dtype(torch.float32)
        
        # 初始化简化类的临时数据存储
        self.temp_data = utils.SimplifyClass(Manager())
        
        # 设定使用的设备
        self.device = device

        # 根据提供的参数初始化特征提取器
        self.feature_ext_args = feature_ext_args
        if feature_ext_args is None:
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                config.get('dataset', 'intermediate'))
        else:
            assert isinstance(feature_ext_args, dict)
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                config.get('dataset', 'intermediate'),
                                                **feature_ext_args)

        # 分割数据集为训练、验证和测试集
        data_saving_path = os.path.join(config.get('dataset', 'intermediate'), 'dataset.idx')
        
        # 检查是否已保存了分割数据，且不需要更新
        if os.path.exists(data_saving_path) and (not self.feature_extractor.update):
            (self.train_dataset, self.validation_dataset, self.test_dataset) = utils.read_pickle(data_saving_path)

            # # 计算良性和恶意apk的数量
            # benign_train = np.sum(self.train_dataset[1] == 0)
            # malicious_train = np.sum(self.train_dataset[1] == 1)

            # benign_val = np.sum(self.validation_dataset[1] == 0)
            # malicious_val = np.sum(self.validation_dataset[1] == 1)

            # benign_test = np.sum(self.test_dataset[1] == 0)
            # malicious_test = np.sum(self.test_dataset[1] == 1)

            # # 打印数据量
            # total_data = len(self.train_dataset[0]) + len(self.validation_dataset[0]) + len(self.test_dataset[0])
            # print(f"总数据量: {total_data}")
            # print(f"训练数据量: {len(self.train_dataset[0])} (良性: {benign_train}, 恶意: {malicious_train})")
            # print(f"验证数据量: {len(self.validation_dataset[0])} (良性: {benign_val}, 恶意: {malicious_val})")
            # print(f"测试数据量: {len(self.test_dataset[0])} (良性: {benign_test}, 恶意: {malicious_test})")

            # 更新数据路径
            def path_tran(data_paths):
                return np.array(
                    [os.path.join(config.get('metadata', 'naive_data_pool'),
                                os.path.splitext(os.path.basename(name))[0] + self.feature_extractor.file_ext) for 
                    name in data_paths])

            self.train_dataset = (path_tran(self.train_dataset[0]), self.train_dataset[1])
            self.validation_dataset = (path_tran(self.validation_dataset[0]), self.validation_dataset[1])
            self.test_dataset = (path_tran(self.test_dataset[0]), self.test_dataset[1])
        else:
            # 预处理恶意软件和良性软件的APK文件，并获取其特征路径
            mal_feature_paths = self.apk_preprocess(config.get('dataset', 'malware_dir'))
            ben_feature_paths = self.apk_preprocess(config.get('dataset', 'benware_dir'))
            feature_paths = mal_feature_paths + ben_feature_paths
            
            # 根据恶意软件和良性软件的数量生成标签
            gt_labels = np.zeros((len(mal_feature_paths) + len(ben_feature_paths)), dtype=np.int32)
            gt_labels[:len(mal_feature_paths)] = 1
            
            # 根据特征路径和标签分割数据
            self.train_dataset, self.validation_dataset, self.test_dataset = self.data_split(feature_paths, gt_labels)
            
            # 保存分割后的数据
            utils.dump_pickle((self.train_dataset, self.validation_dataset, self.test_dataset), data_saving_path)

        # 获取特征词汇表和大小
        self.vocab, _1, _2 = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(self.vocab)
        
        # 获取非API的数量
        self.non_api_size = self.feature_extractor.get_non_api_size(self.vocab)
        
        # 获取类别数量
        self.n_classes = np.unique(self.train_dataset[1]).size


    def data_split(self, feature_paths, labels):
        """
        将数据分为训练、验证和测试集。

        :param feature_paths: 特征文件的路径列表。
        :param labels: 对应的标签列表。
        :return: (训练数据, 训练标签), (验证数据, 验证标签), (测试数据, 测试标签)
        """
        
        # 确保特征文件路径数量与标签数量相同
        assert len(feature_paths) == len(labels)
        
        # 初始化训练、验证和测试集的文件名列表为None
        train_dn, validation_dn, test_dn = None, None, None
        
        # 定义数据集切分文件的路径
        data_split_path = os.path.join(config.get('dataset', 'dataset_dir'), 'tr_te_va_split.name')
        
        # 检查数据切分文件是否存在
        if os.path.exists(data_split_path):
            train_dn, val_dn, test_dn = utils.read_pickle(data_split_path)

        # 如果任何文件名列表为空
        if (train_dn is None) or (validation_dn is None) or (test_dn is None):
            # 从特征文件路径中提取文件名
            data_names = [os.path.splitext(os.path.basename(path))[0] for path in feature_paths]
            
            # 分割数据为训练和测试集，20%为测试集
            train_dn, test_dn = train_test_split(data_names, test_size=0.2, random_state=self.seed, shuffle=True)
            
            # 从训练集中进一步分割出验证集，25%为验证集
            train_dn, validation_dn = train_test_split(train_dn, test_size=0.25, random_state=self.seed, shuffle=True)
            
            # 将切分结果保存为pickle文件
            utils.dump_pickle((train_dn, validation_dn, test_dn), path=data_split_path)

        # 根据提供的文件名列表查询路径
        def query_path(_data_names):
            return np.array(
                [path for path in feature_paths if os.path.splitext(os.path.basename(path))[0] in _data_names])

        # 根据提供的文件名列表查询对应的指示器（布尔列表）
        def query_indicator(_data_names):
            return [True if os.path.splitext(os.path.basename(path))[0] in _data_names else False for path in
                    feature_paths]

        # 查询训练、验证和测试数据的路径
        train_data = query_path(train_dn)
        val_data = query_path(validation_dn)
        test_data = query_path(test_dn)
        
        # 为确保数据与标签一致，随机打乱训练数据和标签
        random.seed(self.seed)
        random.shuffle(train_data)
        train_y = labels[query_indicator(train_dn)]
        random.seed(self.seed)
        random.shuffle(train_y)
        
        # 查询训练、验证和测试数据的标签
        val_y = labels[query_indicator(validation_dn)]
        test_y = labels[query_indicator(test_dn)]
        
        # 返回切分的数据和标签
        return (train_data, train_y), (val_data, val_y), (test_data, test_y)


    def apk_preprocess(self, apk_paths, labels=None, update_feature_extraction=False):
        """
        APK 文件的预处理。
        
        :param apk_paths: APK文件路径列表。
        :param labels: APK文件对应的标签列表，可以为None。
        :param update_feature_extraction: 是否更新特征提取器的状态。
        :return: 处理后的特征路径，和可选的标签。
        """
        
        # 保存特征提取器的当前更新状态
        old_status = self.feature_extractor.update
        
        # 将特征提取器的更新状态设置为提供的参数值
        self.feature_extractor.update = update_feature_extraction
        
        # 如果没有提供标签
        if labels is None:
            # 使用特征提取器从apk_paths中提取特征
            feature_paths = self.feature_extractor.feature_extraction(apk_paths)
            
            # 恢复特征提取器的原始状态
            self.feature_extractor.update = old_status
            
            # 返回特征路径
            return feature_paths
        else:
            # 确保apk文件的数量与标签的数量相匹配
            assert len(apk_paths) == len(labels), \
                '不匹配的数据形状 {} vs. {}'.format(len(apk_paths), len(labels))
            
            # 使用特征提取器从apk_paths中提取特征
            feature_paths = self.feature_extractor.feature_extraction(apk_paths)
            
            labels_ = []
            for i, feature_path in enumerate(feature_paths):
                # 获取不带扩展名的文件名
                fname = os.path.splitext(os.path.basename(feature_path))[0]
                
                # 确保当前文件名在对应的apk路径中
                if fname in apk_paths[i]:
                    # 添加对应的标签到labels_列表中
                    labels_.append(labels[i])
            
            # 恢复特征提取器的原始状态
            self.feature_extractor.update = old_status
            
            # 返回特征路径和对应的标签
            return feature_paths, np.array(labels_)


    def feature_preprocess(self, feature_paths):
        raise NotImplementedError
        # self.feature_extractor.update_cg(feature_paths)


    def feature_api_rpst_sum(self, api_feat_representation_list):
        """
        对API表示进行求和
        :param api_feat_representation_list: 一个稀疏矩阵列表
        """
        
        # 确保输入是一个列表
        assert isinstance(api_feat_representation_list, list), "期望输入是一个列表。"
        
        # 如果列表不为空
        if len(api_feat_representation_list) > 0:
            # 确保列表中的第一个元素是 csr_matrix 类型的稀疏矩阵
            assert isinstance(api_feat_representation_list[0], csr_matrix)
        else:
            # 如果列表为空，则返回一个全为0的矩阵
            return np.zeros(shape=(self.vocab_size - self.non_api_size, self.vocab_size - self.non_api_size),
                            dtype=np.float)
        
        # 将第一个稀疏矩阵转为密集型矩阵，并转换为浮点类型
        adj_array = np.asarray(api_feat_representation_list[0].todense()).astype(np.float32)
        
        # 遍历列表中的其余稀疏矩阵
        for sparse_mat in api_feat_representation_list[1:]:
            # 将稀疏矩阵转为密集型矩阵，转换为浮点类型，并与之前的结果进行相加
            adj_array += np.asarray(sparse_mat.todense()).astype(np.float32)
        
        # 将最终结果中的所有值限制在[0,1]之间
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
        获取输入生产器，返回一个 DataLoader 对象。
        
        :param feature_paths: 特征路径列表。
        :param y: 标签。
        :param batch_size: 每个批次的数据数量。
        :param name: 使用场景名称，默认为'train'。
        :param use_cache: 是否使用缓存，默认为False。
        :return: 返回一个 DataLoader 对象。
        """
        
        # 定义 DataLoader 的参数
        params = {
            'batch_size': batch_size,
            'num_workers': self.feature_ext_args['proc_number'],
            'shuffle': False
        }
        
        # 如果是训练过程，则使用用户设定的缓存值；否则，不使用缓存
        use_cache = use_cache if name == 'train' else False
                
        # 创建 DataLoader，它会使用自定义的 DatasetTorch 数据集对象
        # worker_init_fn 参数用于为每个工作线程设定一个随机种子，确保数据的打乱是随机的
        return torch.utils.data.DataLoader(
            DatasetTorch(feature_paths, y, self, name=name, use_cache=use_cache),
            worker_init_fn=lambda x: np.random.seed(torch.randint(0, 2**31, [1,])[0] + x),
            **params
        )


    def clear_up(self):
        self.temp_data.reset()

    @staticmethod
    def get_modification(adv_x, x, idx, sp=True):
        # 确认adv_x和x是numpy.ndarray类型或torch.Tensor类型的实例
        assert isinstance(adv_x, (np.ndarray, torch.Tensor))
        assert isinstance(x, (np.ndarray, torch.Tensor))
        
        # 计算对抗样本和原始样本之间的差异
        x_mod = adv_x - x
        
        # 根据索引idx选择对应的元素
        if isinstance(x_mod, np.ndarray):
            x_mod = np.array([x_mod[i, idx[i]] for i in range(x.shape[0])])
        else:
            x_mod = torch.stack([x_mod[i, idx[i]] for i in range(x.shape[0])])
            
        # 判断是否需要转为稀疏表示
        if sp:
            # 如果x_mod是torch.Tensor，那么将其转换为稀疏表示并移到cpu上
            # 如果x_mod是numpy.ndarray，那么先将其转换为torch.Tensor，然后转换为稀疏表示并移到cpu上
            if isinstance(x_mod, torch.Tensor):
                return x_mod.to_sparse().cpu().unbind(dim=0)
            else:
                return torch.tensor(x_mod, dtype=torch.int).to_sparse().cpu().unbind(dim=0)
        else:
            # 如果不需要转为稀疏表示，那么直接将其移到cpu上或者分割为numpy数组
            if isinstance(x_mod, torch.Tensor):
                return x_mod.cpu().unbind(dim=0)
            else:
                return np.split(x_mod, x_mod.shape[0], axis=0)


    @staticmethod
    def modification_integ(x_mod_integrated, x_mod):
        # 确认x_mod_integrated和x_mod是列表类型的实例
        assert isinstance(x_mod_integrated, list) and isinstance(x_mod, list)
        
        # 如果x_mod_integrated为空列表，则返回x_mod
        if len(x_mod_integrated) == 0:
            return x_mod
        
        # 确认x_mod_integrated和x_mod的长度相同
        assert len(x_mod_integrated) == len(x_mod)
        
        # 遍历x_mod和x_mod_integrated中的每个元素
        for i in range(len(x_mod)):
            # 确认当前x_mod中的元素不在GPU上，
            # 因为在GPU上的Tensor进行list相加操作的时候是列表拼接，而在CPU上则是张量之间的加法
            assert not x_mod[i].is_cuda
            
            # 更新x_mod_integrated中的元素
            x_mod_integrated[i] += x_mod[i]
            
        # 返回更新后的x_mod_integrated
        return x_mod_integrated



class DatasetTorch(torch.utils.data.Dataset):
    '''为PyTorch定义的数据集'''
    
    def __init__(self, feature_paths, datay, dataset_obj, name='train', use_cache=False):
        '''初始化方法'''
        try:
            # 确保name只有三个允许的值之一
            assert (name == 'train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("仅支持选择：'train'、'val'或'test'.\n")
        
        self.feature_paths = feature_paths  # 特征文件的路径列表
        self.datay = datay                  # 对应的标签数据
        self.dataset_obj = dataset_obj      # 外部提供的数据集对象，提供数据的加载和缓存方法
        self.name = name                    # 数据集名称（例如：'train', 'test', 'val'）
        self.use_cache = use_cache          # 是否使用缓存

    def __len__(self):
        '''返回数据集中的样本总数'''
        return len(self.feature_paths)

    def __getitem__(self, index):
        '''根据索引生成一个数据样本'''
        return self.get_item(index)

    def get_item(self, index):
        '''获取一个数据项'''
        if not self.use_cache:
            # 如果不使用缓存，直接加载并返回数据
            return self.dataset_obj.get_numerical_input(self.feature_paths[index], self.datay[index])
        elif self.dataset_obj.temp_data.is_cached(index):
            # 如果数据已经在缓存中，直接从缓存中返回
            return self.dataset_obj.temp_data.get(index)
        else:
            # 否则，加载数据并将其缓存起来
            feature_vec, label = \
                self.dataset_obj.get_numerical_input(self.feature_paths[index], self.datay[index])
            self.dataset_obj.temp_data.cache(index, feature_vec, label)
            return feature_vec, label

    def reset_memory(self):
        '''重置内部缓存'''
        self.dataset_obj.temp_data.reset()
