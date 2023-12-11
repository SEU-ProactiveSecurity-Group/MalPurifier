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
    # 初始化数据集
    dataset = Dataset(feature_ext_args=args_dict)
    # 获取验证数据和对应的标签
    validation_data, valy = dataset.validation_dataset
    # validation_x1, validation_x2, valy = dataset.get_numerical_input_batch(validation_data, valy)
    
    # 获取验证数据的输入生产器，用于在训练过程中获取批量的输入数据
    val_dataset_producer = dataset.get_input_producer(validation_data, valy, batch_size=2, name='train', use_cache=True)
    import time

    # 对于每个训练周期
    for epoch in range(2):
        # 记录开始时间
        start_time = time.time()

        # 从输入生产器中获取批量的输入数据，并记录索引和对应的输入数据x，以及标签l
        for idx, (x, l) in enumerate(val_dataset_producer):
            import torch
            # 计算每个输入的所有元素之和，并打印出来，同时打印对应的标签
            print(torch.sum(x, dim=-1), l)

        # 打印出本次训练周期花费的时间
        print('cost time:', time.time() - start_time)

    # 清理数据集，释放内存
    dataset.clear_up()



if __name__ == '__main__':
    main_()
