import os, os.path as osp
import cv2
import numpy as np
import zipfile
import json
import pickle
import torch
import dnnlib
import torchvision.transforms as T
import torchvision.transforms.functional_pil as T_pil
import albumentations
import albumentations.augmentations as A

from glob import glob
from PIL import Image, ImageOps
from utils import data_util
from utils.data_util import *

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

def is_image_file(x):
    _, ext = os.path.splitext(x)
    return ext in ['.jpg', '.png', '.webp'] # TODO: add more

#----------------------------------------------------------------------------

class PascalPart(torch.utils.data.Dataset):
    def __init__(self,
        rootdir,
        split,              # ['train', 'val', 'test']
        img_folder,
        seg_folder,
        img_size,           # (height, width)
        max_size    = None,
        transform   = None,   #

    ):
        self.rootdir = rootdir
        self.img_folder = img_folder
        self.seg_folder = seg_folder
        assert split in ['train', 'val', 'test']

        if split == 'test':
            all_indexs = [x for x in sorted(glob(os.path.join(rootdir, self.img_folder, '*'))) if is_image_file(x)]
            all_indexs = [os.path.splitext(os.path.basename(x))[0] for x in all_indexs]
            self.indexs = all_indexs
        else:
            with open(os.path.join(rootdir, f'{split}_set.txt'), 'r') as f:
                self.indexs = [x.strip() for x in f.readlines()]
        if max_size is not None:
            self.indexs = self.indexs[:max_size]

        self.im_width, self.im_height = img_size
        self.transform = transform

        with open(os.path.join(rootdir, 'stats.json'), 'rt') as f:
            stats = json.load(f)
        self.num_class = len(stats['part_class'][0][0])

    @property
    def image_shape(self):
        return [3, self.im_height, self.im_width]

    @property
    def label_shape(self):
        return [self.im_height, self.im_width]

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, i):
        index = self.indexs[i]
        img = Image.open(os.path.join(self.rootdir, self.img_folder, index + '.png'))
        seg = Image.open(os.path.join(self.rootdir, self.seg_folder, index + '.png'))
        assert seg.mode == 'P'

        width, height = img.size
        assert width == seg.size[0] and height == seg.size[1]
        # scale_factor = max(self.dst_width / width, self.dst_height / height)
        # crop_width = int(self.dst_width / scale_factor)
        # crop_height = int(self.dst_height / scale_factor)

        # Center-crop & resize the image
        # resize_kwargs = dict(size=(self.dst_width, self.dst_height),
        #                      box=((width - crop_width)//2, (height - crop_height)//2,
        #                           (width + crop_width)//2, (height + crop_height)//2))
        # img = img.resize(resample=Image.LANCZOS, **resize_kwargs)
        # seg = seg.resize(**resize_kwargs)

        # Resize the image
        img = img.resize(size=(self.im_width, self.im_height))
        seg = seg.resize(size=(self.im_width, self.im_height))

        # Convert to tensor
        img_tensor = torch.from_numpy(np.asarray(img).copy().transpose(2, 0, 1))
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor / 127.5 - 1.  # normalize to [-1, 1]
        seg_tensor = torch.from_numpy(np.asarray(seg).copy())
        return img_tensor, seg_tensor

#----------------------------------------------------------------------------

class CGPart(torch.utils.data.Dataset):
    model_specs = {
        'car':{
            'minivan': '4ef6af15bcc78650bedced414fad522f',
            'truck': '42af1d5d3b2524003e241901f2025878', 
            'suv': '473dd606c5ef340638805e546aa28d99',
            'sedan': '6710c87e34056a29aa69dfdc5532bb13',
            'wagon': 'bad0a52f07afc2319ed410a010efa019',
        },
        'aeroplane': {
            'airliner':'10e4331c34d610dacc14f1e6f4f4f49b',
            'jet':'5cf29113582e718171d03b466c72ce41',
            'fighter':'b0b164107a27986961f6e1cef6b8e434',
            'biplane':'17c86b46990b54b65578b8865797aa0'
        }
    }
    def __init__(self,
        rootdir,
        obj_name,
        dst_shape,          # (width, height)
        transform = None,   #
        length = None,
        indices_path = None,        # txt list containing sampling indices
        classes_version    = None,    # default None: not change anything
        selected_models    = 'all',
        SUN_scene_dir_list  = None,
        bg_prob             = 0.2,
        to_square           = False,
    ):
        assert obj_name in ['car', 'aeroplane']
        self.obj_name = obj_name
        self.to_square = to_square
        # 
        if classes_version is None:
            classes_version = f'CGPart_{obj_name}'
        assert classes_version in [
            f'CGPart_{obj_name}', f'CGPart_{obj_name}_simplified_v0', f'CGPart_{obj_name}_simplified_v1'
        ]

        self.classes_version = classes_version
        self.rootdir = rootdir
        # model selection
        if selected_models == 'all':
            selected_models = list(self.model_specs[obj_name].keys())
        elif isinstance(selected_models, str):
            selected_models = [selected_models]
        
        self.indices = []
        for model_name in selected_models:
            if model_name not in self.model_specs[obj_name].keys():
                raise ValueError(f'Invalid model {model_name}')
            self.indices += [
                os.path.relpath(x, os.path.join(rootdir, 'image'))
                for x in sorted(glob(os.path.join(rootdir, 'image', self.model_specs[obj_name][model_name], '*.png')))
            ]
        # self.indices = [os.path.relpath(x, os.path.join(rootdir, 'image'))
        #                 for x in sorted(glob(os.path.join(rootdir, 'image', '*/*.png')))]

        if indices_path is not None and osp.exists(indices_path):
            indices = np.loadtxt(indices_path, dtype=int)
            self.indices = np.asarray(self.indices)[indices].tolist()
        elif length is not None:
            np.random.shuffle(self.indices)
            self.indices = self.indices[:length] # TODO: configure it
        self.dst_width, self.dst_height = dst_shape
        self.transform = transform

        self.classes = CLASSES[classes_version]
        self.num_class = len(self.classes)
        # set up hook for mask transformation
        if self.classes_version == f'CGPart_{obj_name}':
            self.trans_mask = lambda x: x
        elif self.classes_version == f'CGPart_{obj_name}_simplified_v0':
            self.trans_mask = data_util.__dict__[f'cgpart_{obj_name}_simplify_v0'] 
        elif self.classes_version == f'CGPart_{obj_name}_simplified_v1':
            self.trans_mask = data_util.__dict__[f'cgpart_{obj_name}_simplify_v1'] 

        # SUN used for background augmentation
        if SUN_scene_dir_list is None:
            SUN_scene_dir_list = []
        SUN_dir = 'data/SUN397'
        SUN_scene_dir_list = [
            osp.join(SUN_dir, d)
            for d in SUN_scene_dir_list
        ]
        self.bg_list = []
        for d in SUN_scene_dir_list:
            self.bg_list += list(glob(osp.join(d, '*.jpg')))
        self.bg_list.sort()
        self.bg_prob = bg_prob   # probability that change background

        # pad to square image
        self.resize_w = self.dst_width
        self.resize_h = self.dst_height
        if self.dst_width != self.dst_height and self.to_square:
            self.dst_width = max(self.dst_width, self.dst_height)
            self.dst_height = self.dst_width

    @property
    def image_shape(self):
        return [3, self.dst_height, self.dst_width]

    @property
    def label_shape(self):
        return [self.dst_height, self.dst_width]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        img = Image.open(os.path.join(self.rootdir, 'image', index))
        seg = Image.open(os.path.join(self.rootdir, 'seg', index))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        assert seg.mode == 'P'
        width, height = img.size
        assert width == seg.size[0] and height == seg.size[1]
        # scale_factor = max(self.dst_width / width, self.dst_height / height)
        # crop_width = int(self.dst_width / scale_factor)
        # crop_height = int(self.dst_height / scale_factor)

        # Center-crop & resize the image
        resize_kwargs = dict(size=(self.resize_w, self.resize_h),)
                             # box=((width - crop_width)//2, (height - crop_height)//2,
                             #      (width + crop_width)//2, (height + crop_height)//2))
        img = img.resize(resample=Image.LANCZOS, **resize_kwargs)
        seg = seg.resize(**resize_kwargs)

        # padding to square image (e.g. for car: from 512x384 to 512x512) 
        src_width, src_height = self.resize_w, self.resize_h
        if src_width != src_height and self.to_square:
            # w > h
            if src_width > src_height:
                img = ImageOps.expand(
                    img, 
                    border=(0, (src_width - src_height) // 2, 0, (src_width - src_height) - (src_width - src_height) // 2)
                )
                try:
                    seg = ImageOps.expand(
                        seg, 
                        border=(0, (src_width - src_height) // 2, 0, (src_width - src_height) - (src_width - src_height) // 2)
                    )
                except TypeError:
                    seg = ImageOps.expand(
                        seg.convert('L'), 
                        border=(0, (src_width - src_height) // 2, 0, (src_width - src_height) - (src_width - src_height) // 2)
                    ).convert('P')
            # h > w
            else:
                img = ImageOps.expand(
                    img, 
                    border=((src_height - src_width) // 2, 0, (src_height - src_width) - (src_height - src_width) // 2, 0)
                )
                seg = ImageOps.expand(
                    seg, 
                    border=((src_height - src_width) // 2, 0, (src_height - src_width) - (src_height - src_width) // 2, 0)
                )

        # Add background
        if len(self.bg_list) > 0 and np.random.random() < self.bg_prob:
            bg = Image.open(np.random.choice(self.bg_list))
            bg = bg.resize(resample=Image.LANCZOS, **resize_kwargs)
            img = np.array(img)
            bg = np.asarray(bg)
            seg = np.asarray(seg)
            img[seg == 0] = bg[seg == 0]

        # Convert to tensor
        img_tensor = torch.from_numpy(np.asarray(img).copy().transpose(2, 0, 1))
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor / 127.5 - 1.  # normalize to [-1, 1]
        
        # Convert segmentation
        seg_tensor = torch.from_numpy(self.trans_mask(np.asarray(seg).copy()))
        return img_tensor, seg_tensor

#----------------------------------------------------------------------------

class CGPartReal(torch.utils.data.Dataset):
    def __init__(self,
        rootdir,
        obj_name,
        split,
        short_side,         # (height, width)
        transform = None,   #
        classes_version    = None,    # default None: not change anything
    ):
        assert obj_name in ['car', 'aeroplane']
        self.obj_name = obj_name
        # 
        if classes_version is None:
            classes_version = f'CGPart_{obj_name}'
        assert classes_version in [
            f'CGPart_{obj_name}', f'CGPart_{obj_name}_simplified_v0', f'CGPart_{obj_name}_simplified_v1'
        ]
        self.classes_version = classes_version

        self.rootdir        = rootdir
        self.image_dir      = os.path.join(rootdir, 'Images', f'{obj_name}_imagenet_cropped')
        self.seg_dir        = os.path.join(rootdir, 'Annotations_png', f'{obj_name}_imagenet_cropped')
        imageset_file       = os.path.join(rootdir, 'Image_sets', f'{obj_name}_imagenet_{split}.txt')

        self.indices = np.loadtxt(imageset_file, dtype=str).tolist()
        self.dst_short_side = short_side
        self.transform = transform

        self.classes = CLASSES[classes_version]
        self.num_class = len(self.classes)
        # set up hook for mask transformation
        if self.classes_version == f'CGPart_{obj_name}':
            self.trans_mask = lambda x: x
        elif self.classes_version == f'CGPart_{obj_name}_simplified_v0':
            self.trans_mask = data_util.__dict__[f'cgpart_{obj_name}_simplify_v0'] 
        elif self.classes_version == f'CGPart_{obj_name}_simplified_v1':
            self.trans_mask = data_util.__dict__[f'cgpart_{obj_name}_simplify_v1'] 

    @property
    def image_shape(self):
        return [3, self.dst_height, self.dst_width]

    @property
    def label_shape(self):
        return [self.dst_height, self.dst_width]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        img = Image.open(os.path.join(self.image_dir, index + '.JPEG'))
        seg = Image.open(os.path.join(self.seg_dir, index + '.png'))
        assert seg.mode in ['P', 'L'], seg.mode
        width, height = img.size
        assert width == seg.size[0] and height == seg.size[1]
        # scale_factor = max(self.dst_width / width, self.dst_height / height)
        # crop_width = int(self.dst_width / scale_factor)
        # crop_height = int(self.dst_height / scale_factor)

        # Resize the image so that the long side aligns to dst_long_side
        img = T_pil.resize(img, self.dst_short_side)
        seg = T_pil.resize(seg, self.dst_short_side, interpolation=Image.NEAREST)

        # Convert to tensor
        img_tensor = torch.from_numpy(np.asarray(img).copy().transpose(2, 0, 1))
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor / 127.5 - 1.  # normalize to [-1, 1]
        seg_tensor = torch.from_numpy(self.trans_mask(np.asarray(seg).copy()))
        return img_tensor, seg_tensor

#------------------------------------ DatasetGAN ----------------------------

class DatasetGANDataset(torch.utils.data.Dataset):
    def __init__(self,
        rootdir,
        split,
        obj_name,
        img_size        = (256, 256), # (width, height)
        max_size        = None,
        id_range        = None,
        transform       = None,
        convert_34to8   = False, # convert 34 classes to 8 classes, only applied on face class
        to_square       = False, # convert non-square image to square by padding zeros
    ):
        assert split in ['train', 'test'], f'({split}) is not a qualified split'
        assert obj_name in ['car', 'face', 'cat'], f'({obj_name}) is not a qualified object name'
        self.IMG_EXTS                   = ['png', 'jpg', 'jpeg']
        self.split                      = split
        self.obj_name                   = obj_name
        self.dst_width, self.dst_height = img_size
        self.convert_34to8              = convert_34to8
        self.to_square                  = to_square

        # pad to square image
        if self.dst_width != self.dst_height and self.to_square:
            self.dst_width = max(self.dst_width, self.dst_height)
            self.dst_height = self.dst_width

        self.N_PART_CLASSES = {
            'car': 20, 'cat': 16, 'face': 34
        }
        self.num_class = self.N_PART_CLASSES[obj_name]

        # convert 34 classes to 8 classes
        if self.obj_name == 'face' and self.convert_34to8:
            self.num_class = 8
            
        if split == 'train':
            data_dir = osp.join(rootdir, 'training_data', f'{obj_name}_processed')
        else:
            data_dir = osp.join(rootdir, 'testing_data', f'{obj_name}_{self.N_PART_CLASSES[obj_name]:d}_class')

        # NOTE: DO NOT USE vanilla sorted
        def _get_idx(file_name):
            x, _ = os.path.splitext(os.path.basename(file_name))
            x = x.split('_')[-1]
            return int(x)
        self.img_path_list = sorted([
            osp.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.split('.')[-1].lower() in self.IMG_EXTS
        ], key=_get_idx)
        def _get_idx_mask(file_name):
            x, _ = os.path.splitext(os.path.basename(file_name))
            return int(x.strip('image_mask'))
        self.label_path_list = sorted([
            osp.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('.npy')
        ], key=_get_idx_mask)

        if id_range is not None:
            lo, hi = id_range
            self.img_path_list = self.img_path_list[lo: hi]
            self.label_path_list = self.label_path_list[lo: hi]

        elif (max_size is not None):
            self.img_path_list = self.img_path_list[:max_size]
            self.label_path_list = self.label_path_list[:max_size]

        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    @property
    def image_shape(self):
        return [3, self.dst_height, self.dst_width]

    @property
    def label_shape(self):
        return [self.dst_height, self.dst_width]

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = Image.open(im_path)
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[0, ...]

        # clean up artifacts in the annotation, must do (by DatasetGAN)
        for target in range(1, 50):
            if (lbl == target).sum() < 30:
                lbl[lbl == target] = 0

        if self.obj_name == 'face' and self.convert_34to8:
            lbl = trans_mask(lbl, 'dg_face', 'celebA_8')

        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.preprocess(im, lbl)
        if self.transform is not None:
            im = self.transform(im)
        im = im / 127.5 - 1.  # normalize to [-1, 1]

        return im, lbl

    def preprocess(self, img, lbl):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)

        # padding to square image (e.g. for car: from 512x384 to 512x512) 
        src_width, src_height = self.img_size
        if src_width != src_height and self.to_square:
            # w > h
            if src_width > src_height:
                img = ImageOps.expand(
                    img, 
                    border=(0, (src_width - src_height) // 2, 0, (src_width - src_height) - (src_width - src_height) // 2)
                )
                lbl = ImageOps.expand(
                    lbl, 
                    border=(0, (src_width - src_height) // 2, 0, (src_width - src_height) - (src_width - src_height) // 2)
                )
            # h > w
            else:
                img = ImageOps.expand(
                    img, 
                    border=((src_height - src_width) // 2, 0, (src_height - src_width) - (src_height - src_width) // 2, 0)
                )
                lbl = ImageOps.expand(
                    lbl, 
                    border=((src_height - src_width) // 2, 0, (src_height - src_width) - (src_height - src_width) // 2, 0)
                )

        lbl = torch.from_numpy(np.array(lbl)).long()
        img = torch.from_numpy(np.array(img).transpose(2, 0, 1))
        return img, lbl

#------------------------------------ CelebAMask-HQ -------------------------

class CelebAMaskDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        rootdir,
        imageset_folder = 'celeba_split',
        split           = 'train',
        limit_size      = None,         # number of labeled training images, if None, load all of the data
        img_size        = (256, 256),   # (width, height)
        aug             = False,        # augmentation for training set
        convert_19to8   = True,         # convert 19 classes to 8 classes
    ):
        assert split in ['train', 'val', 'test']
        self.rootdir = rootdir
        self.split = split
        self.limit_size = limit_size
        self.img_size = img_size
        self.convert_19to8 = convert_19to8

        self.num_class = 8 if self.convert_19to8 else 19

        self.idx_list = []

        # if limit_size is None:
        #     self.idx_list = np.loadtxt(os.path.join(imageset_folder, f'{split}.txt'), dtype=int)
        # else:
        #     if os.path.exists(f'data/configs/CelebA_split/{split}_{limit_size:d}_list.txt'):
        #         self.idx_list = np.loadtxt(f'data/configs/CelebA_split/{split}_{limit_size:d}_list.txt', dtype=int)
        #     else:
        #         self.idx_list = np.loadtxt(f'data/configs/CelebA_split/{split}_1500_list.txt', dtype=int)
        #         np.random.shuffle(self.idx_list)
        #         self.idx_list = self.idx_list[:limit_size]
        self.idx_list = np.loadtxt(os.path.join(imageset_folder, f'{split}.txt'), dtype=int)
        if limit_size is not None:
            self.idx_list = self.idx_list[:limit_size]

        self.img_dir = osp.join(self.rootdir, 'CelebA-HQ-img')
        self.label_dir = os.path.join(self.rootdir, 'CelebAMask-HQ-mask-anno')
        self.data_size = len(self.idx_list)

        self.aug = aug
        if aug == True:
            self.aug_t = albumentations.Compose([
                A.transforms.HorizontalFlip(p=0.5),
                A.geometric.transforms.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                    value=0, mask_value=0, p=0.5
                ),
            ])

    def __len__(self):
        return self.data_size
    
    @property
    def image_shape(self):
        return [3, self.img_size[1], self.img_size[0]]

    @property
    def label_shape(self):
        return [self.img_size[1], self.img_size[0]]
    
    def __getitem__(self, idx):
        if idx >= self.data_size:
            idx = idx % (self.data_size)
        img_idx = self.idx_list[idx]
        # Load image
        img_pil = Image.open(osp.join(self.img_dir, f'{img_idx}.jpg')).convert('RGB').resize((self.img_size[0], self.img_size[1]))
        # Load mask
        mask_orig = np.zeros((512, 512), dtype=np.uint8)
        for i_cls, part_cls in enumerate(CLASSES['celebA_19']):
            if part_cls == 'background': continue
            mask_fname = osp.join(self.label_dir, f'{img_idx // 2000:d}', f'{img_idx:05d}_{part_cls}.png')
            if not osp.exists(mask_fname):
                continue
            cur_m = np.asarray(Image.open(mask_fname).convert('L'))
            mask_orig[cur_m > 0] = i_cls
        # transfer 19 classes to 8 classes
        if self.convert_19to8:
            mask_orig = trans_mask(mask_orig, 'celebA_19', 'celebA_8')
        mask_orig = Image.fromarray(mask_orig, mode='L')
        mask_pil = mask_orig.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)

        # perform augmentation
        if self.split == 'train' and self.aug:
            augmented = self.aug_t(image=np.array(img_pil), mask=np.array(mask_pil))
            out_im = Image.fromarray(augmented['image'], mode='RGB')
            out_lbl = np.array(augmented['mask'])
        else:
            out_im = img_pil
            out_lbl = mask_pil
        
        out_im = torch.from_numpy(np.array(out_im).transpose(2, 0, 1))
        out_im = out_im / 127.5 - 1.
        out_lbl = torch.from_numpy(np.array(out_lbl)).long()
        
        return out_im, out_lbl

#----------------------------------------------------------------------------
