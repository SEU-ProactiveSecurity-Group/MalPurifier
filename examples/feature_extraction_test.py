from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

from core.droidfeature import Apk2features
from config import config

cmd_md = argparse.ArgumentParser(description='Arguments for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=4,
                    help='number of threads for features extraction.')
cmd_md.add_argument('--number_of_smali_files', type=int, default=1000000,
                    help='maximum number of produced sequences for each app')
cmd_md.add_argument('--use_top_disc_features', action='store_true', default=True,
                    help='use feature selection or not.')
cmd_md.add_argument('--max_vocab_size', type=int, default=10000,
                    help='maximum number of vocabulary size')
cmd_md.add_argument('--update', action='store_true', default=False,
                    help='update the existed features.')

args = cmd_md.parse_args()


def _main():
    malware_dir_name = config.get('dataset', 'malware_dir')
    print(malware_dir_name)
    benware_dir_name = config.get('dataset', 'benware_dir')
    meta_data_saving_dir = config.get('dataset', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2features(naive_data_saving_dir,
                                     meta_data_saving_dir,
                                     number_of_smali_files=args.number_of_smali_files,
                                     max_vocab_size=args.max_vocab_size,
                                     use_top_disc_features=args.use_top_disc_features,
                                     update=args.update,
                                     proc_number=args.proc_number)
    malware_features = feature_extractor.feature_extraction(malware_dir_name)
    print('The number of malware files: ', len(malware_features))
    benign_features = feature_extractor.feature_extraction(benware_dir_name)
    print('The number of benign files: ', len(benign_features))



if __name__ == '__main__':
    _main()
