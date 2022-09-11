import argparse
import os

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimulatedDataset
from model import Hybrid_ViT, MobileNet_AVG, EfficientNet


@torch.no_grad()
def extract_frame_features(model, transform, args):
    features = []
    img_paths = os.listdir(os.path.join(args.root_path, transform))
    img_paths = [os.path.join(args.root_path, transform, img) for img in img_paths]

    dataset = SimulatedDataset(img_paths)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)

    model.eval()
    bar = tqdm(loader, ncols=120, desc=transform, unit='batch')
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        features.append(feat)
    feature = np.vstack(features)

    if args.pca:
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        feature = pca.apply_py(feature)

    print("normalizing descriptors")
    faiss.normalize_L2(feature)

    feature = torch.from_numpy(feature)

    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)

    torch.save(feature, f'{args.feature_path}/{transform}.pth')

    print(feature.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/mldisk/nfs_shared_/VCDB_simulated/simulated_frame_db')
    parser.add_argument('--feature_path', type=str, default='/hdd/sy/VCDB_simulated/features/test2')
    parser.add_argument('--model', type=str, default='hybrid_vit')
    parser.add_argument('--checkpoint', type=bool, default=False)

    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--pca', type=bool, default=False)
    parser.add_argument('--pca_file', type=str, default='/hdd/sy/DISC/pca/pca_hybrid_vit_256.vt')
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
        checkpoint = '/workspace/data/mobilenet_avg_ep16_ckpt.pth'
    elif args.model == 'desc_1st':
        model = EfficientNet()
        checkpoint = '/workspace/data/iscnet_efficientnet_v2.pth.tar'
    model.to(args.device)

    # Check device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print('before load')  # check weight load
    check_parameters = list(model.named_parameters())[-7]
    print(check_parameters[0], check_parameters[1][:10])

    if args.checkpoint:
        state_dict = torch.load(checkpoint)
        if 'state_dict' in state_dict:
            state_dict = torch.load(checkpoint)['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = torch.load(checkpoint)['model_state_dict']
        elif 'teacher' in state_dict:
            state_dict = torch.load(checkpoint)['teacher']

        if args.model == 'desc_1st':
            model.load_state_dict(state_dict)
        else:
            # model.module.load_state_dict(state_dict)
            model.module.load_state_dict(state_dict, strict=False)

    print('after load')  # check weight load
    check_parameters = list(model.named_parameters())[-7]
    print(check_parameters[0], check_parameters[1][:10])

    # transforms = ['origin', 'BlackBorder_01', 'BlackBorder_02', 'Brightness_01', 'Brightness_02', 'Brightness_03',
    #               'Crop_01', 'Crop_02', 'Flip_H', 'Flip_V', 'GrayScale',
    #               'Logo_01', 'Logo_02', 'Logo_03', 'PIP',
    #               'Rotation_01', 'Rotation_02', 'Rotation_03']

    transforms = ['origin']

    for transform in transforms:
        extract_frame_features(model, transform, args)
