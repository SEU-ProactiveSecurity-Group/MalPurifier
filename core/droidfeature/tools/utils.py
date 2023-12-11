import json
import os
import signal
import warnings
import joblib
import fileinput
import pickle as pkl
import _pickle as cPickle
import gzip

import hashlib
import random
import string
import base64

ENC_KEY = 'cab228a122d3486bac7fab148e8b5aba'

import scipy.sparse as sp
import numpy as np
import torch


def pool_initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """

    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path, file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_list = list()
    for ext in file_exts:
        file_path_list.extend(get_file_name(os.path.join(base_dir, dir_ext), ext))
    # remove duplicate elements
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(file_path_list))


def check_dir(sample_dir):
    """
    check a valid directory and produce a list of file paths
    """
    if isinstance(sample_dir, str):
        if not os.path.exists(sample_dir):
            MSG = "No such directory or file {} exists!".format(sample_dir)
            raise ValueError(MSG)
        elif os.path.isfile(sample_dir):
            sample_path_list = [sample_dir]
        elif os.path.isdir(sample_dir):
            sample_path_list = list(retrive_files_set(sample_dir, "", ".apk|"))
            if len(sample_path_list) <= 0:
                warnings.warn('No files')
        else:
            raise ValueError(" No such path {}".format(sample_dir))
    elif isinstance(sample_dir, list):
        sample_path_list = [path for path in sample_dir if os.path.isfile(path)]
    else:
        MSG = "A directory or a list of paths are allowed!"
        raise ValueError(MSG)

    return sample_path_list


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def load_json(json_path):
    try:
        with open(json_path, 'r') as rh:
            return json.load(rh)
    except IOError as ex:
        raise IOError(str(ex) + ": Unable to load json file.")


def dump_json(obj_dict, file_path):
    try:
        import json
        with open(file_path, 'w+') as fh:
            json.dump(obj_dict, fh)
    except IOError as ex:
        raise IOError(str(ex) + ": Fail to dump dict using json toolbox")


def dump_pickle(data, path, use_gzip=False):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    if not use_gzip:
        with open(path, 'wb') as wr:
            pkl.dump(data, wr)
    else:
        with gzip.open(path, 'wb') as wr:
            pkl.dump(data, wr)
    return True


def read_pickle(path, use_gzip=False):
    if os.path.isfile(path):
        if not use_gzip:
            with open(path, 'rb') as fr:
                return pkl.load(fr)
        else:
            with gzip.open(path, 'rb') as fr:
                return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_pickle_frd_space(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with gzip.open(path, 'wb') as wr:
        cPickle.dump(data, wr)
    return True


def read_pickle_frd_space(path):
    if os.path.isfile(path):
        with gzip.open(path, 'rb') as fr:
            return cPickle.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_list_of_lists(data, path):
    assert isinstance(data[0], list) and isinstance(data, list)
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    with open(path, 'wb') as f:
        for inter_list in data:
            f.write((','.join(inter_list) + '\n').encode())
    return


def read_list_of_lists(path):
    assert os.path.exists(path) and os.path.isfile(path)

    rtn_data = []
    with open(path, 'rb') as f:
        for l in f.readlines():
            rtn_data.append(l.decode('utf-8').strip().split(','))


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def read_txt(path, mode='r'):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
            return lines
    else:
        raise ValueError("{} does not seen like a file path.\n".format(path))


def dump_txt(data_str, path, mode='w'):
    if not isinstance(data_str, str):
        raise TypeError

    with open(path, mode) as f_w:
        f_w.write(data_str)


def read_file_by_fileinput(file_path, inplace=True):
    try:
        return fileinput.input(file_path, inplace=inplace)
    except IOError as ex:
        raise IOError(str(ex))


class SimplifyClass:
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()
        return

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, img, lbl):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (img, lbl)


def build_kwargs(keys, arg_dict):
    st = ''
    for key in keys:
        st += '%s:%s\n' % (key, str(arg_dict[key]))
    return st


def inverse_kwargs(vars):
    assert isinstance(vars, list)
    return dict(var.split(':') for var in vars)


def save_args(fout, args):
    if isinstance(args, str):
        dump_txt(args, fout, mode='w')
    elif isinstance(args, dict):
        args_str = build_kwargs(args.keys(), args)
        dump_txt(args_str, fout, mode='w')
    else:
        raise TypeError("Expected str or dict.")


def load_args(fout):
    if os.path.exists(fout):
        return inverse_kwargs(read_txt(fout))
    else:
        raise FileNotFoundError("No such file {}".format(fout))


def get_group_args(args, args_parser, title):
    import argparse
    assert isinstance(args, argparse.Namespace) and isinstance(args_parser, argparse.ArgumentParser)
    for group in args_parser._action_groups:
        if group.title == title:
            return {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
        else:
            continue
    return


def tensor_coo_sp_to_ivs(sparse_tensor):
    return sparse_tensor._indices(), sparse_tensor._values(), sparse_tensor.size()


def ivs_to_tensor_coo_sp(ivs, device='cpu'):
    return torch.sparse_coo_tensor(ivs[0], ivs[1], ivs[2], device=device)


def sp_to_symmetric_sp(sparse_mx):
    sparse_mx = sparse_mx + sparse_mx.T.multiply(sparse_mx.T > sparse_mx) - sparse_mx.multiply(sparse_mx.T > sparse_mx)
    sparse_eye = sp.csr_matrix(sparse_mx.sum(axis=0) > 1e-8).T.multiply(sp.eye(sparse_mx.shape[0]))
    return (sparse_mx.multiply(1. - np.eye(*sparse_eye.shape))).tocsr() + sparse_eye


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.DoubleTensor(sparse_mx.data)
    return torch.sparse.DoubleTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def to_tensor(feature_x=None, labels=None, device='cpu'):
    """Convert features, labels from array or sparse matrix to
    torch Tensor.
    code is adapted from: https://github.com/deeprobust/DeepRobust/graph/utils.py
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """

    def _to_torch_tensor(mat):
        if sp.issparse(mat):
            mat = sparse_mx_to_torch_sparse_tensor(mat)
        elif isinstance(mat, torch.Tensor):
            pass
        else:
            mat = torch.DoubleTensor(mat)
        return mat

    feature_x = _to_torch_tensor(feature_x).to(device)
    if labels is None:
        return feature_x
    else:
        labels = torch.LongTensor(labels).to(device)
        return feature_x, labels


def to_device(feature_x=None, labels=None, device='cpu'):
    if feature_x is not None:
        assert isinstance(feature_x, torch.Tensor)
        feature_x = feature_x.to(device)
    if labels is not None:
        assert isinstance(labels, torch.Tensor)
        labels = labels.to(device)
    return feature_x, labels


def psn(x_tensor, prob, lower_value=0., upper_value=1.):
    assert 1. >= prob >= 0.
    uni_noises = torch.DoubleTensor(x_tensor.shape).uniform_()
    salt_pos = uni_noises >= prob
    uni_noises[salt_pos] = upper_value
    uni_noises[~salt_pos] = lower_value
    if x_tensor.get_device() >= 0:
        return uni_noises.to(x_tensor.get_device())
    else:
        return uni_noises


class NonnegWeightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = torch.clamp(w, min=0.)
            module.weight.data = w


def round_x(x, alpha=0.5):
    """
    rounds x by thresholding it according to alpha which can be a scalar or vector
    :param x:
    :param alpha: threshold parameter
    :return: a double tensor of 0s and 1s.
    """
    return (x >= alpha).double()


def get_x0(x, rounding_threshold=0.5, is_sample=False):
    """
    Helper function to randomly initialize the the inner maximizer algos
    randomize such that the functionality is preserved.
    Functionality is preserved by maintaining the features present in x

    https://github.com/ALFA-group/robust-adv-malware-detection/

    :param x: training sample
    :param is_sample: flag to sample randomly from feasible area or return just x
    :return: randomly sampled feasible version of x
    """
    if is_sample:
        rand_x = round_x(torch.rand(x.size()), alpha=rounding_threshold)
        if x.is_cuda:
            rand_x = rand_x.cuda()
        return (rand_x.byte() | x.byte()).double()
    else:
        return x


def or_tensors(x_1, x_2):
    """
    ORs two float tensors by converting them to byte and back
    Note that byte() takes the first 8 bit after the decimal point of the float
    e.g., 0.0 ==> 0
          0.1 ==> 0
          1.1 ==> 1
        255.1 ==> 255
        256.1 ==> 0
    Subsequently the purpose of this function is to map 1s float tensors to 1
    and those of 0s to 0. I.e., it is meant to be used on tensors of 0s and 1s.
    :param x_1: tensor one
    :param x_2: tensor two
    :return: float tensor of 0s and 1s.
    """
    return (x_1.byte() | x_2.byte()).double()


def xor_tensors(x_1, x_2):
    """
    XORs two float tensors by converting them to byte and back
    Note that byte() takes the first 8 bit after the decimal point of the float
    e.g., 0.0 ==> 0
          0.1 ==> 0
          1.1 ==> 1
        255.1 ==> 255
        256.1 ==> 0
    Subsequently the purpose of this function is to map 1s float tensors to 1
    and those of 0s to 0. I.e., it is meant to be used on tensors of 0s and 1s.
    :param x_1: tensor one
    :param x_2: tensor two
    :return: float tensor of 0s and 1s.
    """
    return (x_1.byte() ^ x_2.byte()).double()


def get_mal_data(x_batch, y_batch):
    """
    malware feature vectors
    """
    assert isinstance(x_batch, torch.Tensor) and isinstance(y_batch, torch.Tensor)
    mal_x_batch = x_batch[y_batch == 1]
    mal_y_batch = y_batch[y_batch == 1]
    null_flag = len(mal_x_batch) <= 0
    return mal_x_batch, mal_y_batch, null_flag


def get_mal_ben_data(x_batch, y_batch):
    """
    malware feature vectors
    """
    assert isinstance(x_batch, torch.Tensor) and isinstance(y_batch, torch.Tensor)
    mal_x_batch = x_batch[y_batch == 1]
    ben_x_batch = x_batch[y_batch == 0]
    mal_y_batch = y_batch[y_batch == 1]
    ben_y_batch = y_batch[y_batch == 0]
    null_flag = len(mal_x_batch) <= 0
    return mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag
#################################################################################
################################# smali code ####################################
#################################################################################

def java_class_name2smali_name(cls):
    """
       Transform a typical xml format class into smali format

       :param cls: the input class name
       :rtype: string
    """
    if cls is None:
        return
    if not isinstance(cls, str):
        raise ValueError("Expected a string")

    return "L" + cls.replace(".", "/") + ";"


def remove_duplicate(components):
    if isinstance(components, list):
        return ['.'.join(list(filter(None, comp.strip().split('.')))) for comp in components]
    elif isinstance(components, str):
        return '.'.join(list(filter(None, components.strip().split('.'))))
    else:
        raise TypeError("Types of 'list' and 'str' are expected, but got {}.".format(type(components)))


def crypt_identifier(idf, seed=2345):
    if idf == '':
        return ''
    random.seed(seed)

    def md5_transform():
        if isinstance(idf, str):
            return hashlib.md5(idf.encode('utf-8'))
        else:
            return hashlib.md5(idf)

    start_idx = random.choice(range(0, 8))
    length = random.choice(range(8, 16 - start_idx))

    head_letter = random.choice(string.ascii_lowercase)
    return head_letter + md5_transform().hexdigest()[start_idx:start_idx + length]


def random_string(code):
    def sha1_transform():
        if isinstance(code, str):
            return hashlib.sha1(code.encode('utf-8'))
        else:
            return hashlib.sha1(code)

    return random.choice(string.ascii_uppercase) + sha1_transform().hexdigest()[:8]


def string_on_code(code):
    def md5_transform():
        if isinstance(code, str):
            return hashlib.md5(code.encode('utf-8'))
        else:
            return hashlib.md5(code)

    return 'md5' + md5_transform().hexdigest()


def random_name(seed=2345, code='abc'):
    if not isinstance(seed, int):
        raise TypeError("Integer required.", type(seed), seed)
    random.seed(seed)
    sample_letters = [random.sample(string.ascii_letters, 1)[0] for _ in range(12)]
    return random.choice(string.ascii_uppercase) + random_string(code) + ''.join(sample_letters)


def apply_encryption(base_string):
    key = ENC_KEY * int(len(base_string) / len(ENC_KEY) + 1)
    xor_string = ''.join(chr(ord(x) ^ ord(y)) for (x, y) in zip(base_string, key))
    return base64.b64encode(xor_string.encode('utf-8')).decode('utf-8')


def get_sha256(file_path):
    fh = open(file_path, 'rb')
    sha256 = hashlib.sha256()
    while True:
        data = fh.read(8192)
        if not data:
            break
        sha256.update(data)
    return sha256.hexdigest()
