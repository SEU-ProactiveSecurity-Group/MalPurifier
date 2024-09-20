from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from core.defense import Dataset

cmd_md = argparse.ArgumentParser(description='arguments for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=6,
                    help='The number of threads for features extraction.')
cmd_md.add_argument('--number_of_smali_files', type=int, default=1000000,
                    help='The maximum number of produced sequences for each app')
cmd_md.add_argument('--max_vocab_size', type=int, default=10000,
                    help='The maximum number of vocabulary size')
cmd_md.add_argument('--use_top_disc_features', action='store_true', default=True,
                    help='Whether use feature selection or not.')
cmd_md.add_argument('--update', action='store_true', default=False,
                    help='Whether update the existed features.')
args = cmd_md.parse_args()
args_dict = vars(args)


def main_():
    dataset = Dataset(feature_ext_args=args_dict)
    validation_data, valy = dataset.validation_dataset
    
    val_dataset_producer = dataset.get_input_producer(validation_data, valy, batch_size=2, name='train', use_cache=True)
    import time

    for epoch in range(2):
        start_time = time.time()

        for idx, (x, l) in enumerate(val_dataset_producer):
            import torch
            print(torch.sum(x, dim=-1), l)

        print('cost time:', time.time() - start_time)

    dataset.clear_up()



if __name__ == '__main__':
    main_()
