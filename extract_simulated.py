import argparse
import os
import time

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimulatedDataset
from model import Hybrid_ViT, MobileNet_AVG, EfficientNet
from util import load_checkpoint, negative_embedding_subtraction


@torch.no_grad()
def extract_frame_features(model, transform, args):
    features = []
    img_paths = os.listdir(os.path.join(args.root_path, transform))
    img_paths = [os.path.join(args.root_path, transform, img) for img in img_paths]

    dataset = SimulatedDataset(img_paths, img_size=512)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)

    model.eval()
    bar = tqdm(loader, ncols=120, desc=transform, unit='batch')
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        features.append(feat)
    print(f'feature extraction: {time.time() - start:.2f} sec')
    start = time.time()
    feature = np.vstack(features)
    print(f'convert to numpy: {time.time() - start:.2f} sec')

    if args.pca:
        start = time.time()
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        feature = pca.apply_py(feature)
        print(f'apply pca: {time.time() - start:.2f} sec')

    if args.negative_embedding:
        start = time.time()
        print("negative embedding subtraction")
        train = np.load(f'/hdd/sy/DISC/negative_embbeding/train_feats.npy')
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
        feature = negative_embedding_subtraction(feature, _train, index_train, num_iter=1, k=10, beta=0.35)
        print(f'apply neg embedding: {time.time() - start:.2f} sec')
    else:
        start = time.time()
        print("normalizing descriptors")
        faiss.normalize_L2(feature)
        print(f'normalize: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)
    print(f'make dir: {time.time() - start:.2f} sec')

    start = time.time()
    torch.save(feature, f'{args.feature_path}/{transform}.pth')
    print(f'save time: {time.time() - start:.2f} sec')

    print(feature.shape)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/mldisk/nfs_shared_/VCDB_simulated/simulated_frame_db')
    parser.add_argument('--feature_path', type=str, default='/hdd/sy/VCDB_simulated/features/test/desc_1st_neg')
    parser.add_argument('--model', type=str, default='desc_1st')  # hybrid_vit, mobilenet_avg, desc_1st
    parser.add_argument('--checkpoint', type=bool, default=True)

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--pca', type=bool, default=False)   # only hybrid_vit, mobilenet_avg
    parser.add_argument('--pca_file', type=str, default='/hdd/sy/DISC/pca/pca_hybrid_vit_256.vt')

    parser.add_argument('--negative_embedding', type=bool, default=False)   # only desc_1st
    args = parser.parse_args()

    print(args)

    # models
    print("loading model")
    model = None
    checkpoint = None
    if args.model == 'hybrid_vit':
        model = Hybrid_ViT()
        checkpoint = '/workspace/ckpts/res26_vits32_fivr_triplet.pth'
    elif args.model == 'mobilenet_avg':
        model = MobileNet_AVG()
        checkpoint = '/workspace/ckpts/mobilenet_avg_ep16_ckpt.pth'
    elif args.model == 'desc_1st':
        model = EfficientNet(eval_p=1.0)
        checkpoint = '/workspace/ckpts/efficientnet_v2_disc_contrastive.pth.tar'
    model.to(args.device)

    # Check device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print('before load')  # check weight load
    check_parameters = list(model.named_parameters())[-7]
    print(check_parameters[0], check_parameters[1][:10])

    if args.checkpoint:
        load_checkpoint(args, model, checkpoint)

    print('after load')  # check weight load
    check_parameters = list(model.named_parameters())[-7]
    print(check_parameters[0], check_parameters[1][:10])

    # transforms = ['origin', 'BlackBorder_01', 'BlackBorder_02', 'Brightness_01', 'Brightness_02', 'Brightness_03',
    #               'Crop_01', 'Crop_02', 'Flip_H', 'Flip_V', 'GrayScale',
    #               'Logo_01', 'Logo_02', 'Logo_03', 'PIP',
    #               'Rotation_01', 'Rotation_02', 'Rotation_03',
    #               'multi_BrC', 'multi_BrL', 'multi_BrP',
    #               'multi_CL', 'multi_CP', 'multi_LP', 'multi_CLP']

    transforms = ['origin']

    for transform in transforms:
        extract_frame_features(model, transform, args)
