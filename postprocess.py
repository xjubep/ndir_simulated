import argparse
import os
import time

import faiss
import numpy as np
import torch

from util import negative_embedding_subtraction


def apply_neg_embedding(transform, args):
    feature = torch.load(f'{args.feature_path}/{transform}.pth').numpy()
    train = np.load(f'/hdd/sy/DISC/negative_embbeding/train_feats.npy')

    start = time.time()
    index_train = faiss.IndexFlatIP(train.shape[1])
    ngpu = faiss.get_num_gpus()
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False
    index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
    index_train.add(train)

    # DBA on training set
    sim, ind = index_train.search(train, k=10)
    k = 10
    alpha = 3.0
    _train = (train[ind[:, :k]] * (sim[:, :k, None] ** alpha)).sum(axis=1)
    _train /= np.linalg.norm(_train, axis=1, keepdims=True)

    index_train = faiss.IndexFlatIP(train.shape[1])
    ngpu = faiss.get_num_gpus()
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False
    index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
    index_train.add(_train)
    print(f'{time.time() - start:.2f} sec')

    start = time.time()
    feature = negative_embedding_subtraction(feature, _train, index_train, num_iter=1, k=10, beta=0.35)
    feature = torch.from_numpy(feature)
    print(f'{time.time() - start:.2f} sec')

    if not os.path.exists(args.neg_feature_path):
        os.makedirs(args.neg_feature_path)

    torch.save(feature, f'{args.neg_feature_path}/{transform}.pth')
    print(f'[{transform}] finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply negative embedding')
    parser.add_argument('--root_path', type=str, default='/mldisk/nfs_shared_/VCDB_simulated/simulated_frame_db')
    parser.add_argument('--feature_path', type=str, default='/hdd/sy/VCDB_simulated/features/desc_1st_512')
    parser.add_argument('--neg_feature_path', type=str, default='/hdd/sy/VCDB_simulated/features/desc_1st_neg')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--negative_embedding', type=bool, default=True)  # only desc_1st
    args = parser.parse_args()

    print(args)

    transforms = ['origin', 'BlackBorder_01', 'BlackBorder_02', 'Brightness_01', 'Brightness_02', 'Brightness_03',
                  'Crop_01', 'Crop_02', 'Flip_H', 'Flip_V', 'GrayScale',
                  'Logo_01', 'Logo_02', 'Logo_03', 'PIP',
                  'Rotation_01', 'Rotation_02', 'Rotation_03',
                  'multi_BrC', 'multi_BrL', 'multi_BrP',
                  'multi_CL', 'multi_CP', 'multi_LP', 'multi_CLP']

    for transform in transforms:
        apply_neg_embedding(transform, args)
