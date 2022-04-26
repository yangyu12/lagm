import os, os.path as osp
import json
import time
import argparse
import numpy as np

from torchvision import transforms
import torch
import torch.nn.functional as F
import shutil
import torchvision
import dnnlib
import pickle
from tqdm import tqdm
import PIL.Image
from glob import glob
import torch.distributed
from training.dataset import CelebAMaskDataset, PascalPart
from utils.metrics import calc_metrics

#----------------------------------------------------------------------------

def benchmark(args):
    # Setup
    device = torch.device('cuda')

    # Load the checkpoint with highest validation performance
    if os.path.isdir(args.network_path):
        with open(os.path.join(args.network_path, 'metric-segmentation-val_student.jsonl'), 'rt') as f:
            metrics = [json.loads(line) for line in f]
        max_iou_idx = np.asarray([entry['results']['fg_mIoU'] for entry in metrics]).argmax()
        ckpt_pkl = os.path.join(args.network_path, metrics[max_iou_idx]['snapshot_pth'])
    else:
        assert os.path.isfile(args.network_path)
        ckpt_pkl = args.network_path
    print(ckpt_pkl)

    # Construct model
    model_class = {
        'unet': dict(class_name='training.networks.UNet', input_channels=3, output_channels=args.num_class),
        'deeplabv3_101': dict(class_name='training.networks.DeepLabv3', output_channels=args.num_class)
    }
    model_kwargs = model_class[args.arch]
    classifier = dnnlib.util.construct_class_by_name(**model_kwargs).eval().requires_grad_(False).to(device)

    # Load model
    assert osp.exists(ckpt_pkl)
    _, file_ext = os.path.splitext(ckpt_pkl)
    if file_ext == '.pkl':  # Load Phase I network parameters
        with open(ckpt_pkl, 'rb') as f:
            classifier = pickle.load(f)['S'].eval().requires_grad_(False).to(device)
    elif file_ext == '.pth':  # Load Phase II network parameters
        classifier.load_state_dict(torch.load(ckpt_pkl))
        classifier = classifier.eval().to(device)
    else:
        raise NotImplementedError

    # Construct test dataset
    if args.data == 'pascal':
        test_data = PascalPart(rootdir=args.testing_path, split='test', img_folder='CroppedImages',
                               seg_folder='CroppedSegmentation', img_size=args.segres)
    elif args.data == 'celeba':
        test_data = CelebAMaskDataset(rootdir=args.testing_path, split='test', img_size=args.segres)
    else:
        raise NotImplementedError
    test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    # Evaluate on test set
    test_result_dict = calc_metrics(model=classifier, testloader=test_data, device=device, nC=args.num_class)
    miou = test_result_dict['results']['mIoU']
    fg_miou = test_result_dict['results']['fg_mIoU']
    print(f"IOU: {miou * 100}  ", f"fg IOU {fg_miou * 100}  ", f"ckpt: {ckpt_pkl}")

#----------------------------------------------------------------------------

class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_path_list,
        label_path_list,
        trans           = lambda x: x,
        img_size        = (128, 128),
    ):
        self.label_trans = trans
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = PIL.Image.open(im_path)
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(PIL.Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        lbl = self.label_trans(lbl)
        lbl = PIL.Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)
        im = im * 2. - 1.
        return im, lbl, im_path

    def transform(self, img, lbl):
        if self.img_size is not None:
            img = img.resize((self.img_size[0], self.img_size[1]))
            lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=PIL.Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl

#----------------------------------------------------------------------------

def cross_validate(args):
    # Setup
    device = torch.device('cuda')
    if os.path.isdir(args.network_path):
        ckpt_all = glob(os.path.join(args.network_path, "*"))
        cp_list = [data for data in ckpt_all if ('.pkl' in data) or ('.pth' in data) and ('BEST' not in data)]
        cp_list.sort()
        cp_list = cp_list[-20:]  # Only keep last 20 checkpoints to save time
    else:
        assert os.path.isfile(args.network_path)
        cp_list = [args.network_path]
    data_all = glob(os.path.join(args.testing_path, "*"))
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]
    images.sort()
    labels.sort()

    # Construct model
    model_class = {
        'unet': dict(class_name='training.networks.UNet', input_channels=3, output_channels=args.num_class),
        'deeplabv3_101': dict(class_name='training.networks.DeepLabv3', output_channels=args.num_class),
    }
    model_kwargs = model_class[args.arch]
    classifier = dnnlib.util.construct_class_by_name(**model_kwargs).eval().requires_grad_(False).to(device)

    # Launch cross validation
    cross_mIOU = []
    fold_num = int(len(images) / 5)
    for i in range(5):
        # Construct cross validation dataset
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
        print("Val Data length,", str(len(val_image)))
        print("Testing Data length,", str(len(test_image)))
        val_data = ImageLabelDataset(img_path_list=val_image, label_path_list=val_label, img_size=args.segres)
        val_data = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
        test_data = ImageLabelDataset(img_path_list=test_image, label_path_list=test_label, img_size=args.segres)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        # Model selection
        best_miou = 0
        best_val_miou = 0
        for resume in cp_list:
            # Load model
            assert osp.exists(resume)
            _, file_ext = os.path.splitext(resume)
            if file_ext == '.pkl':  # Load Phase I network parameters
                if 'generator.pkl' in resume:
                    continue
                with open(resume, 'rb') as f:
                    classifier = pickle.load(f)['S'].eval().requires_grad_(False).to(device)
            elif file_ext == '.pth':  # Load Phase II network parameters
                classifier.load_state_dict(torch.load(resume))
                classifier = classifier.eval().to(device)
            else:
                raise NotImplementedError

            # Evaluate on validation set
            result_dict = calc_metrics(model=classifier, testloader=val_data, device=device, nC=args.num_class)

            # Evaluate on test set if it is the best
            if result_dict['results']['mIoU'] > best_val_miou:
                best_val_miou = result_dict['results']['mIoU']
                test_result_dict = calc_metrics(model=classifier, testloader=test_data, device=device, nC=args.num_class)
                best_miou = test_result_dict['results']['mIoU']
                print("Best IOU ,", str(best_miou), "ckpt: ", resume)

        cross_mIOU.append(best_miou)

    print(cross_mIOU)
    print("cross validation mean:" , 100 * np.mean(cross_mIOU) )
    print("cross validation std:", 100 * np.std(cross_mIOU))
    result = {"Cross validation mean": 100 * np.mean(cross_mIOU), "Cross validation std": 100 * np.std(cross_mIOU),
              "Cross validation": [100 * x for x in cross_mIOU] }
    with open(os.path.join(args.network_path, 'cross_validation.json'), 'w') as f:
        json.dump(result, f)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', help='The architecture of segmentation network', type=str, default='unet')
    parser.add_argument('--data', help='The dataset class to be evaluated on, pascal or celeba', type=str, default='pascal')
    parser.add_argument('--num_class', help='The number of output channels of segmentation network', type=int, default=8)
    parser.add_argument('--network_path', help='The path to network parameters', type=str)
    parser.add_argument('--segres', help='The input resolution of segmentation network', nargs=2, type=int, default=(256, 256))
    parser.add_argument('--testing_path', help='The path to test image', default=str)
    parser.add_argument('--cv', help='Whether to conduct cross validation evaluation, only support for car20, face34, '
                                     'and cat16', action='store_true')
    args = parser.parse_args()
    print(args)
    if args.cv:
        cross_validate(args)
    else:
        benchmark(args)
