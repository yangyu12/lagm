import os
import json
import copy
import shutil
import argparse
import scipy.io as scio
import numpy as np
from pathlib import Path
import dnnlib
import PIL.Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
import pycocotools.mask as cocomask
from tabulate import tabulate
from utils.visualization import get_palette

# ----------------------------------------------------------------------------
# We assume part semantic segmentation, which means different instances of
# the same part category are not distinguished. For example,
# (i)  different instances of engine, wheel etc.
# (ii) different instances of wing (lwing & rwing)
# We make compact category representation rather than the casual one as in original
# matlab code.

PASCAL_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

PASCAL_PART2ID_ = [
    # [0] aeroplane
    # Ref: [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8903484
    dict(body=1, stern=2, lwing=3, rwing=3, tail=2,
         **{f'engine_{i:d}': 4 for i in range(30)},
         **{f'wheel_{i:d}': 5 for i in range(30)}),
    # [1] bicycle
    dict(fwheel=1, bwheel=1, saddle=2, handlebar=3, chainwheel=4,
         **{f'headlight_{i:d}': 5 for i in range(30)}),
    # [2] bird
    dict(head=1, leye=2, reye=2, beak=3, torso=4, neck=5,
         lwing=6, rwing=6, lleg=7, rleg=7, lfoot=8, rfoot=8, tail=9),
    # [3] boat (only silhouette mask)
    dict(),
    # [4] bottle
    dict(cap=1, body=2),
    # [5] bus
    dict(frontside=1, leftside=2, rightside=3, backside=4, roofside=5,
         leftmirror=6, rightmirror=6, fliplate=7, bliplate=8,
         **{f'door_{i:d}': 9 for i in range(30)},
         **{f'wheel_{i:d}': 10 for i in range(30)},
         **{f'headlight_{i:d}': 11 for i in range(30)},
         **{f'window_{i:d}': 12 for i in range(30)}, ),
    # [6] car (the same set of parts with bus)
    dict(frontside=1, leftside=2, rightside=3, backside=4, roofside=5,
         leftmirror=6, rightmirror=6, fliplate=7, bliplate=8,  # fliplate = front license plate
         **{f'door_{i:d}': 9 for i in range(30)},
         **{f'wheel_{i:d}': 10 for i in range(30)},
         **{f'headlight_{i:d}': 11 for i in range(30)},
         **{f'window_{i:d}': 12 for i in range(30)}, ),
    # [7] cat
    dict(head=1, leye=2, reye=2, lear=3, rear=4, nose=5, torso=6, neck=7, lfleg=8, lfpa=9,
         rfleg=8, rfpa=9, lbleg=8, lbpa=9, rbleg=8, rbpa=9, tail=10, ),
    # [8] chair (only sihouette mask)
    dict(),
    # [9] cow
    dict(head=1, leye=2, reye=2, lear=3, rear=4, muzzle=5, lhorn=6, rhorn=6, torso=7, neck=8,
         lfuleg=9, lflleg=9, rfuleg=9, rflleg=9, lbuleg=9, lblleg=9, rbuleg=9, rblleg=9, tail=10, ),
    # [10] diningtable (only silhouette mask)
    dict(),
    # [11] dog (the same set of parts with cat except including muzzle) TODO
    dict(head=1, leye=2, reye=2, lear=3, rear=4, nose=5, torso=6, neck=7, lfleg=8, lfpa=9,
         rfleg=8, rfpa=9, lbleg=8, lbpa=9, rbleg=8, rbpa=9, tail=10,
         muzzle=11),
    # [12] horse (the same set of parts with cow except excluding lhorn & rhorn and including lfho & rfho & lbho & rbho)
    # Ref: https://arxiv.org/pdf/1505.02438.pdf
    dict(head=1, leye=1, reye=1, lear=1, rear=1, muzzle=1, lfho=3, rfho=3, lbho=3, rbho=3, torso=2, neck=4,
         lfuleg=3, lflleg=3, rfuleg=3, rflleg=3, lbuleg=3, lblleg=3, rbuleg=3, rblleg=3, tail=5, ),
    # [13] motorbike
    dict(fwheel=1, bwheel=2, handlebar=3, saddle=4, **{f'headlight_{i:d}': 5 for i in range(30)}, ),
    # [14] person
    dict(head=1, leye=2, reye=2, lear=3, rear=4, lebrow=5, rebrow=5, nose=6, mouth=7, hair=8,
         torso=9, neck=10, llarm=11, luarm=11, lhand=12, rlarm=13, ruarm=13, rhand=12, llleg=14, luleg=14,
         lfoot=15, rlleg=14, ruleg=14, rfoot=15, ),
    # [15] pottedplant
    dict(pot=1, plant=2, ),
    # [16] sheep (the same set of parts with cow)
    dict(head=1, leye=2, reye=2, lear=3, rear=4, muzzle=5, lhorn=6, rhorn=6, torso=7, neck=8,
         lfuleg=9, lflleg=9, rfuleg=9, rflleg=9, lbuleg=9, lblleg=9, rbuleg=9, rblleg=9, tail=10, ),
    # [17] sofa (only sihouette mask)
    dict(),
    # [18] train
    dict(head=1, hfrontside=2, hleftside=3, hrightside=4, hbackside=5, hroofside=6,
         **{f'headlight_{i:d}': 7 for i in range(30)},
         **{f'coach_{i:d}': 8 for i in range(30)},
         **{f'cfrontside_{i:d}': 9 for i in range(30)},
         **{f'cleftside_{i:d}': 10 for i in range(30)},
         **{f'crightside_{i:d}': 11 for i in range(30)},
         **{f'cbackside_{i:d}': 12 for i in range(30)},
         **{f'croofside_{i:d}': 13 for i in range(30)}, ),
    # [19] tvmonitor
    dict(screen=1),
]

# ----------------------------------------------------------------------------

def annotation_mat2py(data):
    """
    data:
        'anno': ndarray((1, 1), dtype=tuple)
          |-- ndarray((1, ), dtype=str)             ==> file index
          |-- ndarray((1, n), dtype=tuple)
               |-- ndarray((1, ), dtype=str)        ==> category name
               |-- ndarray((1, 1), dtype=uint8)     ==> category idx
               |-- ndarray((h, w), dtype=uint8)     ==> silhouette
               |-- ndarray((1, p), dtype=tuple('part name', mask))      ==> part mask
    """
    pyanno = dnnlib.EasyDict()

    # Convert the mat format data to python EasyDict
    anno_mat = data.pop('anno').item()
    pyanno.file = anno_mat[0].item()
    pyanno.instances = []
    for mat_obj in anno_mat[1][0]:
        pyinst = dnnlib.EasyDict()
        pyinst.class_name = mat_obj[0].item()
        pyinst.class_idx = mat_obj[1].item()
        pyinst.silhouette = mat_obj[2]
        try:
            pyinst.part = [(p[0].item(), p[1]) for p in mat_obj[3][0]]
        except IndexError:
            pyinst.part = []
        pyanno.instances.append(pyinst)

    return pyanno

# ----------------------------------------------------------------------------

def anno2map(anno):
    """
    anno: {
        'file'              ==> file index
        'instances': [
            { 'class_name':
              'class_idx':
              'silhouette':
              'part': [(name, mask), ...]
            },
            ...
        ]
    }
    """
    height, width = anno.instances[0].silhouette.shape
    cls_mask = np.zeros((height, width), dtype=np.uint8)
    inst_mask = np.zeros((height, width), dtype=np.uint8)
    part_mask = np.zeros((height, width), dtype=np.uint8)

    for i, inst in enumerate(anno.instances):
        assert height == inst.silhouette.shape[0] and width == inst.silhouette.shape[1]
        cls_mask[inst.silhouette.astype(np.bool)] = inst.class_idx
        inst_mask[inst.silhouette.astype(np.bool)] = i

        for pname, pmask in inst.part:
            assert pname in PASCAL_PART2ID_[inst.class_idx-1], f'The part {pname} is undefined in {inst.class_name}'
            assert inst.silhouette[pmask.astype(np.bool)].all(), 'The part region is not a complete subregion of the object'
            # if not inst.silhouette[pmask].all():
                # print(f'Warning: [{anno.file}: {pname}] The part region is not a complete subregion of the object')
            pid = PASCAL_PART2ID_[inst.class_idx-1][pname]
            part_mask[pmask.astype(np.bool)] = pid

    return cls_mask, inst_mask, part_mask

# ----------------------------------------------------------------------------

def parse_pascal_part_annotation():
    # Set up
    rootdir = 'data/VOC2010'
    segmat_dir = os.path.join(rootdir, 'part_segmentation', 'Annotations_Part')
    segpng_dir = os.path.join(rootdir, 'part_segmentation', 'png_part')
    segjson_dir = os.path.join(rootdir, 'part_segmentation', 'json_part')
    os.makedirs(segpng_dir, exist_ok=True)
    os.makedirs(segjson_dir, exist_ok=True)
    matfiles = [str(f) for f in sorted(Path(segmat_dir).rglob('*')) if str(f).endswith('.mat') and os.path.isfile(f)]
    palette  = get_palette()

    # Iterate over the mat files
    for matf in tqdm(matfiles):
        mat_data = scio.loadmat(matf)
        pyanno = annotation_mat2py(mat_data)

        # Save the annotation as json file
        json_anno = copy.deepcopy(pyanno)
        for inst in json_anno.instances:
            rle_silh = cocomask.encode(np.asfortranarray(inst.silhouette))
            rle_silh['counts'] = rle_silh['counts'].decode('utf-8')
            inst.silhouette = rle_silh
            for i, (pname, pmask) in enumerate(inst.part):
                rle_pmask = cocomask.encode(np.asfortranarray(pmask))
                rle_pmask['counts'] = rle_pmask['counts'].decode('utf-8')
                inst.part[i] = (pname, rle_pmask)
        with open(os.path.join(segjson_dir, pyanno.file + '.json'), 'w') as f:
            json.dump(json_anno, f)

        # Save the part png
        _, _, seg_im = anno2map(pyanno)
        seg_im = PIL.Image.fromarray(seg_im, mode='P')
        seg_im.putpalette(palette)
        seg_im.save(os.path.join(segpng_dir, pyanno.file + '.png'))

# ----------------------------------------------------------------------------

def construct_pascal_part(args):
    # Source directories
    palette = get_palette()
    srcimg_dir = os.path.join(args.src, 'JPEGImages')
    srcbox_dir = os.path.join(args.src, 'Annotations')
    srcjson_dir = os.path.join(args.src, 'part_segmentation', 'json_part')

    # Destination directories
    # class_name = 'car' # horse, cow, sheep, aero, bus, motor
    # dstdir = f'data/pascal_{class_name}'
    dest_split = 'test' if args.split == 'val' else args.split
    dstimg_dir = os.path.join(args.dest, args.obj, dest_split, 'JPEGImages')
    dstseg_dir = os.path.join(args.dest, args.obj, dest_split, 'SegmentationPart')
    dstbox_dir = os.path.join(args.dest, args.obj, dest_split, 'Annotations')
    os.makedirs(dstimg_dir, exist_ok=True)
    os.makedirs(dstseg_dir, exist_ok=True)
    os.makedirs(dstbox_dir, exist_ok=True)
    npart = max(PASCAL_PART2ID_[PASCAL_CLASS.index(args.obj)].values()) + 1

    stats = dnnlib.EasyDict(
        indexs = [],
        single_class = [],      # bool suggesting if the image only contains one class
        single_instance = [],   # bool suggesting if the image only contains one instance
        part_class = [],        # a np ndarray to suggest the parts that appear in the image
        area = [],              # the proportion of the area occupied by interested object wrt. the whole image
    )

    # Load image list
    image_list_file = os.path.join(os.path.join(args.src, f'ImageSets/Main/{args.obj}_{args.split}.txt'))
    with open(image_list_file, 'rt') as f:
        indices = []
        for line in f.readlines():
            line = line.strip().split(' ')
            if int(line[-1]) == 1:
                indices.append(line[0])

    for index in tqdm(indices):

        # Convert the json to pyanno
        jsonf = os.path.join(srcjson_dir, index + '.json')
        with open(jsonf, 'r') as f:
            json_anno = json.load(f)
        pyanno = copy.deepcopy(json_anno)
        for inst in pyanno['instances']:
            rle_silh = inst['silhouette']
            rle_silh['counts'] = rle_silh['counts'].encode('utf-8')
            inst['silhouette'] = cocomask.decode(rle_silh).astype(np.uint8)
            for i, (pname, rle_pmask) in enumerate(inst['part']):
                rle_pmask['counts'] = rle_pmask['counts'].encode('utf-8')
                pmask = cocomask.decode(rle_pmask).astype(np.uint8)
                inst['part'][i] = (pname, pmask)

        # Convert annotation to part segmentation map
        height, width = pyanno['instances'][0]['silhouette'].shape
        part_mask = np.zeros((height, width), dtype=np.uint8)
        single_cls = True
        single_inst = (len(pyanno['instances']) == 1)

        for i, inst in enumerate(pyanno['instances']):
            assert height == inst['silhouette'].shape[0] and width == inst['silhouette'].shape[1]
            if inst['class_name'] != args.obj:
                single_cls = False
                continue
            for pname, pmask in inst['part']:
                assert pname in PASCAL_PART2ID_[inst['class_idx'] - 1], f'The part {pname} is undefined in {args.obj}'
                assert inst['silhouette'][pmask.astype(np.bool)].all(), 'The part is not a complete subregion of the object'
                pid = PASCAL_PART2ID_[inst['class_idx'] - 1][pname]
                part_mask[pmask.astype(np.bool)] = pid
        # area = float(np.sum(part_mask > 0)) / float(height * width)

        # Skip
        if all([inst['class_name'] != args.obj for inst in pyanno['instances']]):
            continue

        # Make statistics
        stats.indexs.append(pyanno['file'])
        stats.single_class.append(single_cls)
        stats.single_instance.append(single_inst)
        part_class = np.zeros((1, npart))
        part_class[0, np.unique(part_mask)] = 1
        stats.part_class.append(part_class)
        stats.area.append(float(np.sum(part_mask > 0)) / float(height * width))

        # Save data and annotation
        shutil.copyfile(os.path.join(srcimg_dir, pyanno['file'] + '.jpg'),
                        os.path.join(dstimg_dir, pyanno['file'] + '.jpg'))      # Copy image
        shutil.copyfile(os.path.join(srcbox_dir, pyanno['file'] + '.xml'),
                        os.path.join(dstbox_dir, pyanno['file'] + '.xml'))      # Copy annotation
        part_seg = PIL.Image.fromarray(part_mask, mode='P')
        part_seg.putpalette(palette)
        part_seg.save(os.path.join(dstseg_dir, pyanno['file'] + '.png'))        # Save part mask

    # Conclude the statistics
    print('Concluding PASCAL-Part/' + args.obj)
    print('Image folder: ', dstimg_dir)
    print('Segmentation folder: ', dstseg_dir)
    print()
    print('Number of images: ', len(stats.single_class))
    print('Number of parts: ', npart)
    print('Number of single-class images: ', sum(stats.single_class))
    print('Number of single-instance images: ', sum(stats.single_instance))
    area = np.asarray(stats.area)
    print(f'Area of the objects: [{np.min(area)*100:.2f}, {np.max(area)*100:.2f}],'
          f' mean={np.mean(area)*100:.2f}, std={np.std(area)*100:.3f}.')
    part_class = np.concatenate(stats.part_class, axis=0)
    num_per_part = np.sum(part_class, axis=0)
    name_per_part = [
        [k for k, v in PASCAL_PART2ID_[PASCAL_CLASS.index(args.obj)].items() if v==i][:7]
        for i in range(npart)
    ]
    table = [[name, num] for name, num in zip(name_per_part, num_per_part)]
    print()
    print('Number of parts')
    print(tabulate(table))

    # Save statistics
    stats.part_class = [x.tolist() for x in stats.part_class]
    with open(os.path.join(args.dest, args.obj, dest_split, 'stats.json'), 'w') as f:
        json.dump(stats, f)

# ----------------------------------------------------------------------------

def compute_mutual_iou(a, b):
    # (N, 4) (x1, y1, x2, y2)
    a = a[:, None]
    b = b[None]

    inter_box = np.zeros((a.shape[0], b.shape[1], 4), dtype=np.float)
    inter_box[:, :, :2] = np.maximum(a[:, :, :2], b[:, :, :2])
    inter_box[:, :, 2:4] = np.minimum(a[:, :, 2:4], b[:, :, 2:4])
    intersection = np.prod(np.maximum(inter_box[:, :, 2:4] - inter_box[:, :, :2], 0.), axis=2)

    union_box = np.zeros((a.shape[0], b.shape[1], 4), dtype=np.float)
    union_box[:, :, :2] = np.minimum(a[:, :, :2], b[:, :, :2])
    union_box[:, :, 2:4] = np.maximum(a[:, :, 2:4], b[:, :, 2:4])
    union = np.prod(np.maximum(union_box[:, :, 2:4] - union_box[:, :, :2], 0.), axis=2)

    mutual_iou = intersection / (union + 1e-8)
    np.fill_diagonal(mutual_iou, 0.)
    return mutual_iou

# ----------------------------------------------------------------------------

def extend_box(box, im_h, im_w, mode='square'):
    x1, y1, x2, y2 = box
    x1 = x1 - 1
    y1 = y1 - 1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    box_h = y2 - y1
    box_w = x2 - x1

    # infer target scale
    scale = max(box_h, box_w)

    # infer target center
    cx_hi = max(im_w / 2, im_w - box_w / 2)
    cx_lo = min(im_w / 2, box_w / 2)
    cy_hi = max(im_h / 2, im_h - box_h / 2)
    cy_lo = min(im_h / 2, box_h / 2)
    cx = min(max(cx, cx_lo), cx_hi)
    cy = min(max(cy, cy_lo), cy_hi)

    # recompute x1, y1, x2, y2
    clip_w = lambda x: min(max(x, 0), im_w)
    clip_h = lambda x: min(max(x, 0), im_h)
    x1 = clip_w(cx - scale / 2)
    x2 = clip_w(cx + scale / 2)
    y1 = clip_h(cy - scale / 2)
    y2 = clip_h(cy + scale / 2)
    return int(x1), int(y1), int(x2), int(y2)

# ----------------------------------------------------------------------------

def crop_with_padding(im, box):
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    box_w = x2 - x1
    patch_scale = max(box_h, box_w)
    if im.ndim == 3:
        im_patch = np.zeros((patch_scale, patch_scale, 3), dtype=np.uint8)
    else:
        im_patch = np.zeros((patch_scale, patch_scale), dtype=np.uint8)
    px1 = int((patch_scale - box_w) / 2)
    px2 = px1 + box_w
    py1 = int((patch_scale - box_h) / 2)
    py2 = py1 + box_h
    im_patch[py1: py2, px1: px2] = im[y1: y2, x1: x2]
    return im_patch

# ----------------------------------------------------------------------------

def prepare_images_and_segmentations(args):
    palette = get_palette()

    # Destination directories
    dstimg_dir = os.path.join(args.dest, args.obj, args.split, 'JPEGImages')
    dstseg_dir = os.path.join(args.dest, args.obj, args.split, 'SegmentationPart')
    dstbox_dir = os.path.join(args.dest, args.obj, args.split, 'Annotations')
    npart = max(PASCAL_PART2ID_[PASCAL_CLASS.index(args.obj)].values()) + 1

    # output directories
    outimg_dir = os.path.join(args.dest, args.obj, args.split, 'CroppedImages')
    outseg_dir = os.path.join(args.dest, args.obj, args.split, 'CroppedSegmentation')
    os.makedirs(outimg_dir, exist_ok=True)
    os.makedirs(outseg_dir, exist_ok=True)

    stats = dnnlib.EasyDict(
        indexs=[],
        part_class=[],  # a np ndarray to suggest the parts that appear in the image
        area=[],  # the proportion of the area occupied by interested object wrt. the whole image
    )

    # Load image list
    image_file_list = sorted(glob(os.path.join(dstimg_dir, '*.jpg')))
    indices = [os.path.splitext(os.path.basename(x))[0] for x in image_file_list]

    for index in tqdm(indices):
        # Parse bounding box annotations
        box_anno_file = os.path.join(dstbox_dir, index + '.xml')
        with open(box_anno_file) as f:
            tree = ET.parse(f)
        objects = []
        for obj in tree.findall("object"):
            if obj.find("name").text != args.obj:
                continue

            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ] # 1-based
            objects.append(obj_struct)

        # Compute mutual IoU
        all_boxes = np.asarray([x['bbox'] for x in objects], dtype=np.float) # (N, 4)
        mutual_iou = compute_mutual_iou(all_boxes, all_boxes)

        # Load image and segmentation
        im = PIL.Image.open(os.path.join(dstimg_dir, index + '.jpg'))
        im = np.asarray(im)  # (height, width, 3)
        seg = PIL.Image.open(os.path.join(dstseg_dir, index + '.png'))
        seg = np.asarray(seg)  # (height, width)

        # Iterate over all the boxes
        height = int(tree.find('size').find('height').text)
        width = int(tree.find('size').find('width').text)
        for inst_idx, obj in enumerate(objects):
            # skip uninterested objects
            if obj['name'] != args.obj:
                print(f'discard due to uninterest')
                continue

            # discard overlapping objects
            if np.any(mutual_iou[inst_idx] > 0.05):
                print(f'discard due to overlapping')
                continue

            # discard small objects
            x1, y1, x2, y2 = obj['bbox']
            if (x2 - x1 + 1 < args.minpx) or (y2 - y1 + 1 < args.minpx):
                print(f'discard due to small objects')
                continue

            # extend box to square box or 4:3 box
            adj_box = extend_box(obj['bbox'], height, width)

            # Crop the image and seg
            im_patch = crop_with_padding(im, adj_box)
            seg_patch = crop_with_padding(seg, adj_box)

            # discard empty segmentation
            area = np.mean(np.asarray(seg_patch > 0, dtype=np.float))
            if area < 0.01:
                print(f'discard small segmentation {area:.4f}')
                continue

            # Make statistics
            stats.indexs.append(f'{index}_{inst_idx:d}')
            part_class = np.zeros((1, npart))
            part_class[0, np.unique(seg_patch)] = 1
            stats.part_class.append(part_class)
            stats.area.append(area)

            # Save data and annotation
            PIL.Image.fromarray(im_patch, mode='RGB').save(os.path.join(outimg_dir, f'{index}_{inst_idx:d}.png'))
            seg_patch = PIL.Image.fromarray(seg_patch, mode='P')
            seg_patch.putpalette(palette)
            seg_patch.save(os.path.join(outseg_dir, f'{index}_{inst_idx:d}.png'))  # Save part mask

    # Conclude the statistics
    print('Concluding PASCAL-Part/' + args.obj)
    print('Image folder: ', dstimg_dir)
    print('Segmentation folder: ', dstseg_dir)
    print()
    print('Number of images: ', len(stats.indexs))
    print('Number of parts: ', npart)
    area = np.asarray(stats.area)
    print(f'Area of the objects: [{np.min(area) * 100:.2f}, {np.max(area) * 100:.2f}],'
          f' mean={np.mean(area) * 100:.2f}, std={np.std(area) * 100:.3f}.')
    part_class = np.concatenate(stats.part_class, axis=0)
    num_per_part = np.sum(part_class, axis=0)
    name_per_part = [
        [k for k, v in PASCAL_PART2ID_[PASCAL_CLASS.index(args.obj)].items() if v == i][:7]
        for i in range(npart)
    ]
    table = [[name, num] for name, num in zip(name_per_part, num_per_part)]
    print()
    print('Number of parts')
    print(tabulate(table))

    # Save statistics
    stats.part_class = [x.tolist() for x in stats.part_class]
    with open(os.path.join(args.dest, args.obj, args.split, 'cropped_stats.json'), 'w') as f:
        json.dump(stats, f)

    # Create the train/val split
    if args.split == 'train':
        all_files = sorted(glob(os.path.join(outimg_dir, '*.png')))
        with open(os.path.join(args.dest, args.obj, args.split, 'train_set.txt'), 'wt') as f:
            for filename in all_files[:180]:
                index, _ = os.path.splitext(os.path.basename(filename))
                f.write(index + '\n')
        with open(os.path.join(args.dest, args.obj, args.split, 'val_set.txt'), 'wt') as f:
            for filename in all_files[180:]:
                index, _ = os.path.splitext(os.path.basename(filename))
                f.write(index + '\n')

# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', help='Preprocess the part annotations', action='store_true')

    parser.add_argument('--obj', help='Which class to be constructed', type=str)
    parser.add_argument('--split', help='Which split to be used', type=str)
    parser.add_argument('--src', help='Root directory to the pascal part datasets', type=str)
    parser.add_argument('--dest', help='Destination directory to save the results', type=str)
    parser.add_argument('--crop', help='Whether to crop image', action='store_true')
    parser.add_argument('--minpx', help='The minimum number of pixels', type=int, default=32)
    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    if args.pre:
        parse_pascal_part_annotation()
        exit()

    if not args.crop:
        construct_pascal_part(args)
    else:
        prepare_images_and_segmentations(args)

# ----------------------------------------------------------------------------
