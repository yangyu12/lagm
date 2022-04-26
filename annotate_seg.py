import os, os.path as osp
import sys
import re
import click
import time
import copy
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import pickle
import json
import PIL.Image
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from utils.visualization import get_palette

import dnnlib
from utils import metrics

#----------------------------------------------------------------------------

def report_metrics(result_dict, run_dir=None, snapshot_pkl=None, desc='test'):
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pth=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-segmentation-{desc}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

def setup_snapshot_image_label_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32) // 2 # Half the number for label
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data
    data = list(zip(*[training_set[i] for i in grid_indices]))
    images = data[0]
    labels = data[1]
    return (gw, gh), np.stack([x.numpy() for x in images]), np.stack([x.numpy() for x in labels])

#----------------------------------------------------------------------------

def save_image_label_grid(img, label, fname, drange, grid_size, conf_threshold=0.9):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    # Colorize label
    if label.ndim == 4:
        label_ = label.argmax(axis=1)
        conf_ = label.max(axis=1)
        label = label_
        hc_label = label.copy()
        hc_label[conf_ < conf_threshold] = 255
    else:
        hc_label = None
    cmap = get_palette()
    label = cmap[label]
    if hc_label is not None:
        hc_label = cmap[hc_label]

    gw, gh = grid_size
    _N, C, H, W = img.shape
    assert C == 3, f'{C}'
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    label = label.reshape(gh, gw, H, W, C)
    label = label.transpose(0, 2, 1, 3, 4)
    if hc_label is not None:
        hc_label = hc_label.reshape(gh, gw, H, W, C)
        hc_label = hc_label.transpose(0, 2, 1, 3, 4)

    # Save image
    img = np.concatenate([img, label], axis=1) if hc_label is None else \
        np.concatenate([img, label, hc_label], axis=1)
    img = img.reshape(gh * H * (2 + (hc_label is not None)), gw * W, C)
    assert C == 3
    PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def train_annotator(
    run_dir                 = '.',  # Output directory.
    resume_path             = '',   # Resume path of model
    training_set_kwargs     = {},  # Options for training set.
    validation_set_kwargs   = {},  # Options for validation set.
    # data_loader_kwargs      = {},  # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},  #
    A_kwargs                = {},  # Options for annotator network.
    S_kwargs                = {},  # Options for student network.
    T_kwargs                = {},
    matcher_kwargs          = {},
    gmparam_regex           = '.*',
    A_opt_kwargs            = {},  # Options for annotator optimizer.
    S_opt_kwargs            = {},  # Options for student optimizer.
    loss_kwargs             = {},  # Options for loss function.
    random_seed             = 0,  # Global random seed.
    image_per_batch         = 2, #
    A_interval              = 1,
    S_interval              = 1,
    total_iters             = 150000,  # Total length of the training, measured in thousands of real images.
    niter_per_tick          = 20,  # Progress snapshot interval.
    label_snapshot_ticks    = 200,  # How often to save (image, label) snapshots? None = disable.
    network_snapshot_ticks  = 200,  # How often to save annotator network snapshots? None = disable.
    cudnn_benchmark         = True,  # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,  # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None, # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda')
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    # TODO: not sure if we need the following things
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load labeled dataset
    print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, seed=random_seed)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=image_per_batch, **data_loader_kwargs))
    train_evalset = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    train_eval_loader = torch.utils.data.DataLoader(dataset=train_evalset, batch_size=image_per_batch, drop_last=False,
                                                    **data_loader_kwargs)

    val_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, drop_last=False,
                                             **data_loader_kwargs)

    print()
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)
    print()

    # Load the synthetic data generator
    G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(device)
    assert G.c_dim == 0, 'Only support unconditional generation.'
    G(torch.empty([image_per_batch, G.z_dim], device=device), torch.empty([image_per_batch, G.c_dim], device=device))
    A_kwargs.input_shapes = G.f_shape
    A_kwargs.output_channels = S_kwargs.output_channels = training_set.num_class

    # Save G
    g_data = dict(G_kwargs=dict(G_kwargs), G=copy.deepcopy(G).eval().requires_grad_(False).cpu())
    g_pkl = os.path.join(run_dir, f'generator.pkl')
    with open(g_pkl, 'wb') as f:
        pickle.dump(g_data, f)
    del g_data

    # Construct networks: annotator & student & transform
    print('Constructing networks...')
    if osp.exists(resume_path):
        print(f"Loading models from {resume_path}")
        with open(resume_path, 'rb') as f:
            network_snapshot = pickle.load(f)
        A = network_snapshot['A'].train().requires_grad_(False).to(device)
        S = network_snapshot['S'].train().requires_grad_(False).to(device)
        del network_snapshot
    else:
        # common_kwargs = dict(img_resolution=training_set.resolution, img_channels=training_set.num_channels)
        A = dnnlib.util.construct_class_by_name(**A_kwargs).train().requires_grad_(False).to(device)
        S = dnnlib.util.construct_class_by_name(**S_kwargs).train().requires_grad_(False).to(device)
    T = dnnlib.util.construct_class_by_name(**T_kwargs)

    # Print network summary tables.
    print(A)
    z = torch.empty([image_per_batch, G.z_dim], device=device)
    c = torch.empty([image_per_batch, G.c_dim], device=device)
    img, features = misc.print_module_summary(G, [z, c])
    y = misc.print_module_summary(A, [features])
    img, _ = T(img, y)
    misc.print_module_summary(S, [img])
    del z, c, img, features, y

    # Setup training phases
    # We basically have two phases
    # phase A: optimize the annotator by matching the gradients
    # pahse S: optimize the student by minimizing the task loss
    print('Parameters of segmentation networks to be gradient matched: ')
    for name, param in S.named_parameters():
        if re.fullmatch(gmparam_regex, name):
            print(name, '\t\t', param.shape)
    params_for_gm = [param for name, param in S.named_parameters() if re.fullmatch(gmparam_regex, name)]
    print('Setting up training phases...')
    matcher = dnnlib.util.construct_class_by_name(**matcher_kwargs)
    modules = dict(G=G, A=A, S=S, T=T, matcher=matcher, params_for_gm=params_for_gm)
    loss = dnnlib.util.construct_class_by_name(device=device, **modules, **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, interval in [('A', A, A_opt_kwargs, A_interval), ('S', S, S_opt_kwargs, S_interval)]:
        opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
        phases += [dnnlib.EasyDict(name=name, module=module, opt=opt, interval=interval)]
    for phase in phases:
        phase.start_event = torch.cuda.Event(enable_timing=True)
        phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample labels.
    print('Exporting sample images...')
    grid_size, images, labels = setup_snapshot_image_label_grid(training_set=training_set)
    save_image_label_grid(images, labels, os.path.join(run_dir, 'reals.png'), drange=[-1, 1], grid_size=grid_size)
    grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(image_per_batch)
    grid_c = torch.from_numpy(labels).to(device).split(image_per_batch)
    images, labels = [], []
    for z, c in zip(grid_z, grid_c):
        img, feat = G(z, c)
        lbl = torch.nn.functional.softmax(A(feat), dim=1)
        img, lbl = T(img, lbl)
        images.append(img.cpu())
        labels.append(lbl.cpu())
    images = torch.cat(images).numpy()
    labels = torch.cat(labels).numpy()
    save_image_label_grid(images, labels, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)
    del img, lbl, images, labels

    # Initialize logs.
    print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    train_eval_stats_metrics = {}
    val_stats_metrics = {}
    stats_tfevents = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
    try:
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(run_dir)
    except ImportError as err:
        print('Skipping tfevents export:', err)

    # Train
    print(f'Training for {total_iters} iterations...')
    print()
    cur_niter = 0
    cur_tick = 0
    # set cur_niter and cur_tick if models are reloaded from resumed ones
    if osp.exists(resume_path):
        try:
            cur_niter = int(resume_path.split('.')[0][-6:])
            cur_tick = cur_niter // niter_per_tick
        except:
            print(f'Fail to resume iters and ticks, which are zeroed out in the following process')

    tick_start_niter = cur_niter
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    if progress_fn is not None:
        progress_fn(0, total_iters)
    for phase in phases:
        phase.module.requires_grad_(True)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            data = next(training_set_iterator)
            if len(data) == 2:
                lbl_image, mannual_label = data
            elif len(data) == 3:
                lbl_image, mannual_label, _ = data
            else:
                raise ValueError('Dataset format error')
            lbl_image = lbl_image.to(device, dtype=torch.float32)
            mannual_label = mannual_label.to(device, dtype=torch.long)

        # Execute training phases.
        for phase in phases:
            # Initialize gradient
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)

            # Requires grad for parameters to match gradients
            if phase.name == 'A':
                S.train()
                for name, param in S.named_parameters():
                    if re.fullmatch(gmparam_regex, name):
                        param.requires_grad_(True)
            phase.module.train().requires_grad_(True)

            # forward
            syn_z = torch.randn([image_per_batch, G.z_dim], device=device)
            syn_c = torch.zeros([image_per_batch, G.c_dim], device=device)
            loss.forward(phase=phase.name, lbl_img=lbl_image, manual_label=mannual_label, syn_z=syn_z, syn_c=syn_c)

            # Backward & update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
            phase.module.eval().requires_grad_(False)

        # Update state.
        cur_niter += 1

        # Perform maintenance tasks once per tick.
        done = (cur_niter >= total_iters)
        if (not done) and (cur_tick != 0) and (cur_niter < tick_start_niter + niter_per_tick):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"niter {training_stats.report0('Progress/niter', cur_niter):<8d}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/iter {training_stats.report0('Timing/sec_per_iter', (tick_end_time - tick_start_time) / (cur_niter - tick_start_niter)):<7.2f}"]
        training_stats.report0('Timing/maintenance_sec', maintenance_time)
        training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30)
        training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30)
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            print()
            print('Aborting...')

        # Save image-label snapshot.
        if (label_snapshot_ticks is not None) and (done or cur_tick % label_snapshot_ticks == 0):
            images, labels = [], []
            for z, c in zip(grid_z, grid_c):
                img, feat = G(z, c)
                lbl = torch.nn.functional.softmax(A(feat), dim=1)
                img, lbl = T(img, lbl)
                images.append(img.cpu())
                labels.append(lbl.cpu())
            images = torch.cat(images).numpy()
            labels = torch.cat(labels).numpy()
            save_image_label_grid(images, labels, os.path.join(run_dir, f'fakes{cur_niter:06d}.png'),
                                  drange=[-1, 1], grid_size=grid_size)
            del images, labels, img, lbl

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('A', A), ('S', S)]:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_niter:06d}.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

        # Evaluate metrics on trainset.
        if snapshot_data is not None:   
            # Evaluate the performance of student on labeled data
            # only validate on small dataset
            if len(train_eval_loader.dataset) <= 500:
                print('Evaluation of student on training set...')
                result_dict = metrics.calc_metrics(model=snapshot_data['S'], testloader=train_eval_loader, device=device)
                report_metrics(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl, desc='train_student') #
                train_eval_stats_metrics.update(result_dict["results"])

            # Evaluate the performance of student on labeled data
            print('Evaluation of student on tset set...')
            result_dict = metrics.calc_metrics(model=snapshot_data['S'], testloader=val_loader, device=device)
            report_metrics(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl, desc='val_student') #
            val_stats_metrics.update(result_dict["results"])
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_niter)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in train_eval_stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/train/{name}', value, global_step=global_step, walltime=walltime)
            for name, value in val_stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/val/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_niter, total_iters)

        # Update state.
        cur_tick += 1
        tick_start_niter = cur_niter
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    print()
    print('Exiting...')

#----------------------------------------------------------------------------

def load_kwargs(
    cfg_file,

    opt                 = None,
    batch               = None, #
    alr                 = None, # Learning rate for updating annotator
    aint                = None, # The interval to update annotator
    slr                 = None, # Learning rate for updating annotator
    sint                = None, # The interval to update student
    steps               = None,
    resume_path         = None, # The path of resume path of train dir

    gmparam             = None,
):
    # Load base configuration
    args = json.load(open(cfg_file, 'r'))
    args = dnnlib.EasyDict(args)
    for k, v in args.items():
        if isinstance(v, dict):
            args[k] = dnnlib.EasyDict(v)
    desc = os.path.basename(cfg_file)[:-5]  # .strip('.json') # [0]

    #
    args.matcher_kwargs = dnnlib.EasyDict(class_name="training.loss.GradientMatcher")

    # ------------------------------------------
    # Optimization options
    # ------------------------------------------

    if opt is not None:
        assert isinstance(opt, str)
        if not opt == 'sgd':
            desc += f'-{opt}opt'
        if opt == 'adam':
            args.A_opt_kwargs.class_name = args.S_opt_kwargs.class_name = 'torch.optim.Adam'
            args.A_opt_kwargs.pop('momentum')
            args.S_opt_kwargs.pop('momentum')

    if batch is not None:
        assert isinstance(batch, int) and batch > 0
        args.image_per_batch = batch
        desc += f'-batch{batch}'

    if alr is not None:
        assert isinstance(alr, float) and alr > 0.
        args.A_opt_kwargs.lr = alr
        desc += f'-alr{alr:g}'

    if aint is not None:
        assert isinstance(aint, int) and aint > 0.
        args.A_interval = aint
        desc += f'-aint{aint:g}'

    if slr is not None:
        assert isinstance(slr, float) and slr > 0.
        args.S_opt_kwargs.lr = slr
        desc += f'-slr{slr:g}'

    if sint is not None:
        assert isinstance(sint, int) and sint > 0
        args.S_interval = sint
        desc += f'-sint{sint:g}'

    if steps is not None:
        assert isinstance(steps, int) and steps > 0
        args.total_iters = steps
        desc += f'-{steps}steps'

    if resume_path is not None:
        args.resume_path = resume_path
        desc += f'-resume'

    # ------------------------------------------
    # Gradient matching options
    # ------------------------------------------

    gmparam_specs = {
        # For U-Net
        'down': 'down.*',
        'up': 'up.*',
        'inc': 'inc.*',
        'downup12': 'down[1-2].*|up[1-2].*',
        'downup34': 'down[3-4].*|up[3-4].*',
        'conv': '.*conv.*',
        'bn': '.*bn.*',
        'outc': 'outc.*',
        'u1': 'outc.*|down1.*|up4.*',
        'u2': 'outc.*|down2.*|up3.*',
        # For DeepLab
        'head': '.*classifier.*',
        'r5head': '.*backbone.layer4.*|.*classifier.*',
        'r45head': '.*backbone.layer[3-4].*|.*classifier.*',
        'r345head': '.*backbone.layer[2-4].*|.*classifier.*',
        'r2345head': '.*backbone.layer[1-4].*|.*classifier.*',
    }
    if gmparam is not None:
        assert gmparam in gmparam_specs
        desc += f'-match_{gmparam}'
        args.gmparam_regex = gmparam_specs[gmparam]

    return desc, args

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')

# Config file
@click.option('--cfg', 'cfg_file', help='The config file', type=str)

# Optimization options
@click.option('--opt', help='What kind of optimizer to be used', type=str)
@click.option('--batch', help='The batch size', type=int)
@click.option('--alr', help='The learning rate to update annotator', type=float)
@click.option('--aint', help='The interval to update annotator', type=int)
@click.option('--slr', help='The learning rate to update student', type=float)
@click.option('--sint', help='The interval to update student', type=int)
@click.option('--steps', help='The number of steps', type=int)
@click.option('--resume-path', help='The path of network snapshot for resume', type=str)

@click.option('--gmparam', help='The parameters to be gradient matched', type=str)

# Misc
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

def main(ctx, outdir, dry_run, **config_kwargs):
    dnnlib.util.Logger(should_flush=True)

    run_desc, args = load_kwargs(**config_kwargs)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f"Training data:      {args.training_set_kwargs.rootdir}")
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)
    with open(os.path.join(args.run_dir, 'command.sh'), 'wt') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    # Launch processes.
    print('Launching processes...')
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    train_annotator(**args)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
