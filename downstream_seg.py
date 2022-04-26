import argparse
import json
import logging
import os, os.path as osp
import pickle
import re
import shutil
import sys
import tempfile
import time
import torch
import torch.distributed
import torch.nn.functional as F
import torchvision
import dnnlib
import numpy as np

from glob import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from training.transform import wide_crop_fn
from utils import metrics
from utils.data_util import trans_car_mask_20to12, trans_face_mask_19to8, trans_face_mask_34to8
from utils.visualization import get_palette

#----------------------------------------------------------------------------

def set_logging(log_path):
    """
    Set screen and file logging for root logger
    """
    # get root logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s: %(pathname)s-%(lineno)d - %(levelname)s\n%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set error traceback logged
    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    return logger

#----------------------------------------------------------------------------

def generate_dataset(
    rank,
    generator_file,
    annotator_file,
    run_dir,
    num             = 10000,
    wide_crop       = False
):  
    device = torch.device(f'cuda:{rank:d}')
    # Construct generator & annotator
    with open(generator_file, 'rb') as f:
        G_kwargs = pickle.load(f).pop('G_kwargs')
        print(G_kwargs)
        G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(device)
    
    if isinstance(annotator_file, str):
        annotator_file = [annotator_file]
    is_single_classifier = len(annotator_file) == 1
    annotators = torch.nn.ModuleList([
        pickle.load(open(ann_path, 'rb'))['A']
        for ann_path in annotator_file
    ]).eval().requires_grad_(False).to(device)

    # Prepare archive for storing the generated data
    data_archive = osp.join(run_dir, 'synthetic_data')
    shutil.rmtree(data_archive, ignore_errors=True)
    os.makedirs(data_archive, exist_ok=True)
    os.makedirs(os.path.join(data_archive, 'img'), exist_ok=True)
    os.makedirs(os.path.join(data_archive, 'seg'), exist_ok=True)
    cmap = get_palette()

    logging.info('Start to generate dataset ...')
    batch_size = 4
    bar = range(rank * num // batch_size, (rank + 1) * num // batch_size)
    if rank == 0:
        bar = tqdm(bar)
    for batch_idx in bar:
        # Fetch data
        with torch.no_grad():
            z = torch.randn(batch_size, G.z_dim, device=device)
            c = torch.zeros(batch_size, G.c_dim, device=device)
            img, features = G(z, c)
            labels = F.softmax(annotators[0](features), dim=1)      # (B, C, H, W)
            if wide_crop:
                img, labels = wide_crop_fn(img, labels)
            confidence, gt_seg = labels.max(dim=1)  # (B, H, W)

        # Store the data
        for idx in range(batch_size):
            idx_str = f'{batch_idx * batch_size + idx:08d}'
            img_fname = os.path.join(data_archive, f'img/{idx_str}.png')
            seg_fname = os.path.join(data_archive, f'seg/{idx_str}.png')
            cur_img = (img[idx].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(dtype=torch.uint8, device='cpu')
            cur_label = gt_seg[idx].to(dtype=torch.uint8, device='cpu')

            # Save the image & label as an uncompressed PNG.
            cur_img = Image.fromarray(cur_img.numpy(), 'RGB')
            cur_img.save(img_fname, format='png', compress_level=0, optimize=False)
            cur_label = Image.fromarray(cur_label.numpy(), 'P')
            cur_label.putpalette(cmap)
            cur_label.save(seg_fname, format='png', compress_level=0, optimize=False)

    return None, data_archive

#----------------------------------------------------------------------------

class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self,
        rootdir,
        img_size            = (128, 128),
        max_number          = None,
        mask_transfer_func  = None,  # mask transfer func
    ):
        self.rootdir = rootdir
        self.img_size = img_size
        self.label_trans = mask_transfer_func

        self.indices = [os.path.basename(f.strip('.png')) for f in sorted(glob(os.path.join(rootdir, 'img/*.png')))]
        if max_number is not None:
            self.indices = self.indices[:max_number]
        self.img_filename = os.path.join(rootdir, 'img/%s.png')
        self.lbl_filename = os.path.join(rootdir, 'seg/%s.png')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        im_file = self.img_filename % self.indices[index]
        im = Image.open(im_file)

        lbl_file = self.lbl_filename % self.indices[index]
        lbl = np.asarray(Image.open(lbl_file))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        if self.label_trans is not None:
            lbl = self.label_trans(lbl)
        lbl = Image.fromarray(lbl.astype('uint8'))

        # Resize & ToTensor
        im, lbl = self.transform(im, lbl)
        im = im * 2. - 1.
        return im, lbl

    def transform(self, img, lbl):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl

#----------------------------------------------------------------------------

def train_on_dataset(
    rank: int, 
    model: torch.nn.Module, 
    n_classes: int,
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader    = None,
    writer: SummaryWriter                       = None,
    run_dir: str                                = None,
    is_distributed                              = True,
    total_epochs                                = 20,
    lr                                          = 0.001,
    start_epoch                                 = 0,
    global_steps                                = 0,
    display_every_n_iters                       = 100,
    validate_every_n_epochs                     = 1,
):
    device = torch.device(f'cuda:{rank:d}')
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)

    if rank == 0:
        model.eval()
        with torch.no_grad():
            result_dict = metrics.calc_metrics(model=model, testloader=test_loader, device=device, nC=n_classes)
        model.train()
        results = result_dict['results']
        logging.info(
            f'Test Epoch {start_epoch - 1:>3d} / {total_epochs:>3d} '
            f'mIoU {100.0 * results["mIoU"]:.3f}% '
            f'pACC {100.0 * results["pACC"]:.3f}% '
        )

    # Train the model on the synthetic data with automatic label
    for epoch in range(start_epoch, total_epochs):
        model = model.requires_grad_(True).train().to(device)
        # make shuffling work properly across multiple epochs
        if is_distributed:
            assert isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler)
            train_loader.sampler.set_epoch(epoch)
        for iter, (img, label) in enumerate(train_loader):
            img = img.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            # Forward & backward & update
            pred = model(img)
            if not isinstance(pred, torch.Tensor):
                pred = pred['out']
            loss = torch.nn.functional.cross_entropy(pred, label, ignore_index=255)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Record
            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), global_steps)

            if rank == 0:
                if (iter + 1) % display_every_n_iters == 0 or iter + 1 == len(train_loader) or iter == 0:
                    logging.info(
                        f'Train Epoch {epoch:>3d} / {total_epochs:>3d} '
                        f'Iter {iter + 1:>6d} / {len(train_loader):>6d} '
                        f'Loss {loss.item():.3f} '
                    )

            global_steps += 1

        if rank == 0 and ((epoch + 1) % validate_every_n_epochs == 0 or epoch + 1 == total_epochs):
            assert test_loader is not None
            model.requires_grad_(False).eval()
            result_dict = metrics.calc_metrics(model=model, testloader=test_loader, device=device, nC=n_classes)
            model.requires_grad_(True).train()
            results = result_dict['results']
            if writer is not None:
                writer.add_scalar('Test/mIoU', results['mIoU'], global_steps)
                writer.add_scalar('Test/pACC', results['pACC'], global_steps)
            logging.info(
                f'Test Epoch {epoch:>3d} / {total_epochs:>3d} '
                f'mIoU {100.0 * results["mIoU"]:.3f}% '
                f'pACC {100.0 * results["pACC"]:.3f}% '
            )
            timestamp = time.time()
            jsonl_line = json.dumps(dict(result_dict, timestamp=timestamp)) 
            if run_dir is not None and osp.isdir(run_dir):
                fname = f'metric-evaluate-synthetic-data.jsonl'
                with open(osp.join(run_dir, fname), 'at') as f:
                    f.write(jsonl_line + '\n')

            # Save executor snapshot
            if run_dir is not None and osp.isdir(run_dir):
                if is_distributed:
                    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    torch.save(model.module.state_dict(), osp.join(run_dir, f'segmentator-{epoch:03d}.pth'))
                else:
                    torch.save(model.module.state_dict(), osp.join(run_dir, f'segmentator-{epoch:03d}.pth'))
                logging.info(f'Saving to {osp.join(run_dir, f"segmentator-{epoch:03d}.pth")}')
    return model


#----------------------------------------------------------------------------

def main(rank, args):
    # Reset logging since it's in different process
    if rank == 0:
        set_logging(osp.join(args.run_dir, 'log.log'))
        writer = SummaryWriter(args.run_dir)
    else:
        writer = None
    device = torch.device('cuda', rank)
    random_seed = 0

    # Prepare dataset
    if not osp.exists(args.trainset):
        args.max_size, args.trainset = generate_dataset(rank, args.generator, args.annotator, args.run_dir,
                                                        wide_crop=(args.val_data=='car'), num=10000 // args.num_gpus)
    if args.num_gpus > 1:
        torch.distributed.barrier()
    training_set = ImageLabelDataset(args.trainset, img_size=args.segres, max_number=args.max_size)
    training_set_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=training_set, rank=rank, num_replicas=args.num_gpus, seed=random_seed
    )
    train_loader = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler if args.num_gpus > 1 else None,
                                               batch_size=args.batch_size // args.num_gpus, pin_memory=True,
                                               num_workers=3, prefetch_factor=2)

    if rank == 0:
        logging.info(f'The size of test set: {len(training_set)}')
   
    # Construct validation loader
    if rank == 0:
        if args.val_data in ['car', 'cat', 'face']:
            dataset_kwargs = dnnlib.EasyDict(
                class_name='training.dataset.DatasetGANDataset', rootdir='./data/DatasetGAN/annotation',
                split='test', obj_name=args.val_data, img_size=args.segres, convert_34to8=False,
            )
            valset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
            valset_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, drop_last=False,
                                                        pin_memory=True, num_workers=3, prefetch_factor=2)
            logging.info(f'The size of test set: {len(valset)}')
        else:
            raise NotImplementedError
    else:
        valset_loader = None
    
    # Construct segmentation network
    model_class = {
        'unet': dict(class_name='training.networks.UNet', input_channels=3, output_channels=args.num_class),
        'deeplabv3_101': dict(class_name='training.networks.DeepLabv3', output_channels=args.num_class),
    }
    model_kwargs = model_class[args.arch]
    model = dnnlib.util.construct_class_by_name(**model_kwargs).train().requires_grad_(True).to(device)

    # Train on dataset
    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)
    train_on_dataset(
        rank, model, args.num_class,
        train_loader, valset_loader, writer, 
        run_dir=args.run_dir,
        is_distributed=(args.num_gpus > 1),
    )

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    # dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Execute training loop.
    main(rank, args)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Downstream segmentation')
    parser.add_argument('--outdir', help='The output directory', type=str, default='output/downstream/')
    parser.add_argument('--generator', help='The path to the generator file', type=str, default='')
    parser.add_argument('--annotator', help='The path to the annotator file', type=str, default='')
    parser.add_argument('--arch', help='The architecture of segmentation network', type=str, default='unet')
    parser.add_argument('--num_class', help='The number of output channels of segmentation network', type=int, default=20)
    parser.add_argument('--segres', help='The input resolution of segmentation network', nargs=2, type=int,
                        default=(256, 256))
    parser.add_argument('--trainset', help='If specified, generation process will be skipped', type=str, default='')
    parser.add_argument('--max_size', help='Maximum size of trainset', type=int, default=None)
    parser.add_argument('--val_data', help='', type=str, default='car')
    parser.add_argument('--batch_size', help='The batch size', type=int, default=8)
    parser.add_argument('--num_gpus', help='The number of GPUs to be paralleled', type=int, default=2)
    args = parser.parse_args()

    # Pick output directory.
    prev_run_dirs = []
    if osp.isdir(args.outdir):
        prev_run_dirs = [x for x in os.listdir(args.outdir) if osp.isdir(osp.join(args.outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = osp.join(args.outdir, f'{cur_run_id:05d}')
    assert not osp.exists(args.run_dir)

    # Create output directory.
    os.makedirs(args.run_dir, exist_ok=True)
    with open(osp.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)
    with open(osp.join(args.run_dir, 'command.sh'), 'wt') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    # Set logging
    set_logging(osp.join(args.run_dir, 'log.log'))

    # Print options.
    logging.info(
        'Training options: \n'
        f'{json.dumps(vars(args), indent=2)} \n\n'
        f'Output directory:   {args.run_dir} \n'
    )

    # Launch processes.
    logging.info('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
