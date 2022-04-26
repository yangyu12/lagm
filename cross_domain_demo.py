import argparse
import pickle
import dnnlib
import torch
import json
from glob import glob
import os
import numpy as np
import PIL.Image
from tqdm import tqdm
from utils.visualization import get_palette
from utils.data_util import cgpart_car_simplify_v0 as trans_mask

#----------------------------------------------------------------------------

def wide_crop_tensor(x):
    B, C, H, W = x.shape
    CH = int(H * 3 // 4)
    return x[:, :, (H - CH) // 2 : (H + CH) // 2]

#----------------------------------------------------------------------------

def main(args):
    # Setup
    device = torch.device(f'cuda')
    network_path = os.path.dirname(args.network_pth)
    palette = get_palette(args.palette_name)

    # Load generator & annotator
    generator_file = os.path.join(network_path, "generator.pkl")
    with open(generator_file, 'rb') as f:
        G_kwargs = pickle.load(f).pop('G_kwargs')
        print(G_kwargs)
        G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(device)
    with open(args.network_pth, 'rb') as f:
        A = pickle.load(f)['A'].eval().requires_grad_(False).to(device)

    # Visualize labeled data
    outlabel_dir = os.path.join(args.outdir, 'labeled_data')
    os.makedirs(outlabel_dir, exist_ok=True)
    indices = [os.path.relpath(x, os.path.join(args.label_path, 'image'))
                   for x in sorted(glob(os.path.join(args.label_path, 'image', '*/*.png')))]
    np.random.shuffle(indices)
    indices = indices[:args.num_vis]
    print('Saving labeled images')
    for idx, index in enumerate(tqdm(indices)):
        img = PIL.Image.open(os.path.join(args.label_path, 'image', index))
        model_id, base_name = index.split('/')
        seg = PIL.Image.open(os.path.join(args.label_path, 'seg', model_id, model_id + base_name[6:]))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        assert seg.mode == 'P'
        seg = trans_mask(np.asarray(seg).copy())
        seg = PIL.Image.fromarray(seg, mode='P')
        seg.putpalette(palette)
        seg = seg.convert('RGB')
        img = np.concatenate([np.asarray(img), np.asarray(seg)], axis=0)
        img = PIL.Image.fromarray(img, 'RGB')
        img_fname = os.path.join(outlabel_dir, f'{idx:04d}.png')
        img.save(img_fname, format='png', compress_level=0, optimize=False)

    # Generate and visualize
    outgen_dir = os.path.join(args.outdir, 'generated_data')
    os.makedirs(outgen_dir, exist_ok=True)
    for idx in tqdm(range(args.num_vis)):
        # Fetch data
        with torch.no_grad():
            z = torch.randn(1, G.z_dim, device=device)
            c = torch.zeros(1, G.c_dim, device=device)
            img, features = G(z, c)
            if args.wide_crop:
                img = wide_crop_tensor(img)
            img = img[0].permute(1, 2, 0)  # (H, W, C)
            seg = A(features)  # (B, C, H, W)
            if args.wide_crop:
                seg = wide_crop_tensor(seg)
            _, seg = torch.max(seg, dim=1)  # (nA, H, W)

        # Save the visualization
        img_fname = os.path.join(outgen_dir, f'{idx:04d}.png')
        img = (img * 127.5 + 128).clamp(0, 255).to(dtype=torch.uint8, device='cpu').numpy()
        seg = PIL.Image.fromarray(seg[0].to(dtype=torch.uint8, device='cpu').numpy(), 'P')
        seg.putpalette(palette)
        seg = np.asarray(seg.convert('RGB'))
        img = np.concatenate([img, seg], axis=0)
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(img_fname, format='png', compress_level=0, optimize=False)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pth', help='The model file to be resumed', type=str)
    parser.add_argument('--label_path', help='The path to labeled data', type=str)
    parser.add_argument('--num_vis', help='The number of checkpoints to be visualized', type=int, default=200)
    parser.add_argument('--palette_name', help='The palette name for visualization', type=str)
    parser.add_argument('--outdir', help='The output directory to save the visualization results', type=str, default='./output/paper_plots/cross_domain_demo')
    parser.add_argument('--wide-crop', help='Whether to crop the generated images/segmentation into wide size', action='store_true')
    args = parser.parse_args()
    main(args)

