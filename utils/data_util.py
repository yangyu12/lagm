import numpy as np
import torch

CLASSES = {
    'dg_car_20': ['background', 'back_bumper', 'bumper', 'car_body', 'car_lights', 'door', 'fender','grilles','handles',
                'hoods', 'licensePlate', 'mirror','roof', 'running_boards', 'tailLight','tire', 'trunk_lids','wheelhub', 'window', 'windshield'],
    'dg_car_12': ['background', 'car_body', 'head light', 'tail light', 'licence plate',
                  'wind shield', 'wheel', 'door', 'handle' , 'wheelhub', 'window', 'mirror'],
    'CGPart_car': [
        'background', 'back_bumper', 'back_left_door', 'back_left_wheel', 'back_left_window', 
        'back_license_plate', 'back_right_door', 'back_right_wheel', 'back_right_window', 'back_windshield',
        'front_bumper', 'front_left_door', 'front_left_wheel', 'front_left_window', 'front_license_plate',
        'front_right_door', 'front_right_wheel', 'front_right_window', 'front_windshield', 'hood',
        'left_frame', 'left_head_light', 'left_mirror', 'left_quarter_window', 'left_tail_light', 
        'right_frame', 'right_head_light', 'right_mirror', 'right_quarter_window', 'right_tail_light', 
        'roof', 'trunck',
    ],  # 32 classes
    'CGPart_car_simplified_v0': [
        'background', 'back_bumper', 'car_body', 'door', 'front_bumper', 
        'head_light', 'hood', 'licence_plate', 'mirror', 'roof', 
        'tail_light', 'trunck', 'wheel', 'window', 'windshield', 
    ],  # 15 classes
    'CGPart_car_simplified_v1': [
        'background', 'car_body', 'door', 
        'head_light', 'licence_plate', 'mirror', 
        'tail_light', 'wheel', 'window', 'windshield', 
    ],  # 10 classes
    'CGPart_aeroplane': [
        'background', 'propeller', 'cockpit', 'wing_left', 'wing_right', 'fin',
        'tailplane_left', 'tailplane_right', 'wheel_front', 'landing_gear_front', 'wheel_back_left',
        'gear_back_left', 'wheel_back_right', 'gear_back_right', 'engine_left', 'engine_right',
        'door_left', 'door_right', 'bomb_left', 'bomb_right', 'window_left',
        'window_right', 'body'
    ],  # 22 classes
    'PascalPart_aeroplane': [
        'background', 'body', 'stern', 'wing', 'engine', 'wheel'
    ],  # 6 classes
    'CGPart_aeroplane_simplified_v0': [
        'background', 'body', 'engine', 'fin',
        'tailplane', 'wheel', 'wing', 
    ],  # 7 classes
    'dg_cat': [f'{i:02d}' for i in range(16)],
    'dg_face': ['background', 'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix',
              'head***ear***lobule', 'head***eye***bottom lid', 'head***eye***eyelashes', 'head***eye***iris',
              'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
              'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
              'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
              'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
              'head***philtrum', 'head***temple', 'head***wrinkles'],
    'celebA_8': ['background', 'ear', 'eye', 'eyebrow', 'skin', 'hair', 'mouth', 'nose'],
    'celebA_19': ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
}

# https://github.com/nv-tlabs/datasetGAN_release/blob/9de77f727d147a4f380469ba098515b999433306/utils/data_util.py#L211
def trans_car_mask_20to12(mask):
    "Transfer car_20 to car_12"
    if isinstance(mask, np.ndarray):
        final_mask = np.zeros(mask.shape)
    elif isinstance(mask, torch.Tensor):
        final_mask = torch.zeros_like(mask)
    final_mask[(mask != 0)] = 1 # car                   everything > carbody
    final_mask[(mask == 4)] = 2 # head light
    final_mask[(mask == 14)] = 5 # tail light
    final_mask[(mask == 10)] = 3 # licence plate
    final_mask[(mask == 19)] = 8 # wind shield
    final_mask[(mask == 15)] = 6 # wheel
    final_mask[(mask == 5)] = 9 # door
    final_mask[(mask == 8)] = 10 # handle
    final_mask[(mask == 17)] = 11 # wheelhub
    final_mask[(mask == 18)] = 7 # window
    final_mask[(mask == 11)] = 4 # mirror
    return final_mask

def trans_face_mask_34to8(mask):
    "Transfer face_34 to face_8"
    src_classes = CLASSES['dg_face']
    tgt_classes = CLASSES['celebA_8']
    if isinstance(mask, np.ndarray):
        tgt_mask = np.zeros_like(mask)
    elif isinstance(mask, torch.Tensor):
        tgt_mask = torch.zeros_like(mask)
    # init
    tgt_mask[mask != src_classes.index('background')] = tgt_classes.index('skin')
    # remove neck
    tgt_mask[mask == src_classes.index('head***neck')] = tgt_classes.index('background')
    # ear
    tgt_mask[mask == src_classes.index('head***ear')] = tgt_classes.index('ear')
    tgt_mask[mask == src_classes.index('head***ear***helix')] = tgt_classes.index('ear')
    tgt_mask[mask == src_classes.index('head***ear***lobule')] = tgt_classes.index('ear')
    # eye
    tgt_mask[mask == src_classes.index('head***eye***bottom lid')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***eyelashes')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***iris')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***pupil')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***sclera')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***tear duct')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('head***eye***top lid')] = tgt_classes.index('eye')
    # eyebrow
    tgt_mask[mask == src_classes.index('head***eyebrow')] = tgt_classes.index('eyebrow')
    # hair
    tgt_mask[mask == src_classes.index('head***hair')] = tgt_classes.index('hair')
    tgt_mask[mask == src_classes.index('head***hair***sideburns')] = tgt_classes.index('hair')
    # mouth
    tgt_mask[mask == src_classes.index('head***mouth***inferior lip')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('head***mouth***oral comisure')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('head***mouth***superior lip')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('head***mouth***teeth')] = tgt_classes.index('mouth')
    # nose
    tgt_mask[mask == src_classes.index('head***nose')] = tgt_classes.index('nose')
    tgt_mask[mask == src_classes.index('head***nose***ala of nose')] = tgt_classes.index('nose')
    tgt_mask[mask == src_classes.index('head***nose***bridge')] = tgt_classes.index('nose')
    tgt_mask[mask == src_classes.index('head***nose***nose tip')] = tgt_classes.index('nose')
    tgt_mask[mask == src_classes.index('head***nose***nostril')] = tgt_classes.index('nose')
    return tgt_mask

def trans_face_mask_19to8(mask):
    "Transfer face_19 to face_8"
    src_classes = CLASSES['celebA_19']
    tgt_classes = CLASSES['celebA_8']
    if isinstance(mask, np.ndarray):
        tgt_mask = np.zeros_like(mask)
    elif isinstance(mask, torch.Tensor):
        tgt_mask = torch.zeros_like(mask)
    # init
    tgt_mask[mask != src_classes.index('background')] = tgt_classes.index('skin')
    # remove unnecessary part (despite glasses (eye_g) since it has overlap with skin)
    tgt_mask[mask == src_classes.index('hat')] = tgt_classes.index('background')
    tgt_mask[mask == src_classes.index('ear_r')] = tgt_classes.index('background')
    tgt_mask[mask == src_classes.index('neck_l')] = tgt_classes.index('background')
    tgt_mask[mask == src_classes.index('neck')] = tgt_classes.index('background')
    tgt_mask[mask == src_classes.index('cloth')] = tgt_classes.index('background')
    # common parts
    tgt_mask[mask == src_classes.index('nose')] = tgt_classes.index('nose')
    tgt_mask[mask == src_classes.index('l_eye')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('r_eye')] = tgt_classes.index('eye')
    tgt_mask[mask == src_classes.index('l_brow')] = tgt_classes.index('eyebrow')
    tgt_mask[mask == src_classes.index('r_brow')] = tgt_classes.index('eyebrow')
    tgt_mask[mask == src_classes.index('l_ear')] = tgt_classes.index('ear')
    tgt_mask[mask == src_classes.index('r_ear')] = tgt_classes.index('ear')
    tgt_mask[mask == src_classes.index('mouth')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('u_lip')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('l_lip')] = tgt_classes.index('mouth')
    tgt_mask[mask == src_classes.index('hair')] = tgt_classes.index('hair')

    return tgt_mask

def cgpart_car_simplify_v0(mask):
    src_classes = CLASSES['CGPart_car']
    tgt_classes = CLASSES['CGPart_car_simplified_v0']
    if isinstance(mask, np.ndarray):
        tgt_mask = np.zeros_like(mask)
    elif isinstance(mask, torch.Tensor):
        tgt_mask = torch.zeros_like(mask)
    tgt_mask[mask != src_classes.index('background')] = tgt_classes.index('car_body')
    
    for front_back in ['back', 'front']:
        tgt_mask[mask == src_classes.index(f'{front_back}_bumper')] = tgt_classes.index(f'{front_back}_bumper')
        tgt_mask[mask == src_classes.index(f'{front_back}_license_plate')] = tgt_classes.index('licence_plate')
        tgt_mask[mask == src_classes.index(f'{front_back}_windshield')] = tgt_classes.index('windshield')

        for left_right in ['left', 'right']:
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_door')] = tgt_classes.index('door')
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_wheel')] = tgt_classes.index('wheel')
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_window')] = tgt_classes.index('window')

    tgt_mask[mask == src_classes.index('hood')] = tgt_classes.index('hood')
    tgt_mask[mask == src_classes.index('roof')] = tgt_classes.index('roof')
    tgt_mask[mask == src_classes.index('trunck')] = tgt_classes.index('trunck')

    for left_right in ['left', 'right']:
        # tgt_mask[mask == src_classes.index(f'{left_right}_frame')] = tgt_classes.index('car_body')
        tgt_mask[mask == src_classes.index(f'{left_right}_head_light')] = tgt_classes.index('head_light')
        tgt_mask[mask == src_classes.index(f'{left_right}_mirror')] = tgt_classes.index('mirror')
        tgt_mask[mask == src_classes.index(f'{left_right}_quarter_window')] = tgt_classes.index('window')
        tgt_mask[mask == src_classes.index(f'{left_right}_tail_light')] = tgt_classes.index('tail_light')
    
    return tgt_mask

def cgpart_car_simplify_v1(mask):
    src_classes = CLASSES['CGPart_car']
    tgt_classes = CLASSES['CGPart_car_simplified_v1']
    if isinstance(mask, np.ndarray):
        tgt_mask = np.zeros_like(mask)
    elif isinstance(mask, torch.Tensor):
        tgt_mask = torch.zeros_like(mask)
    tgt_mask[mask != src_classes.index('background')] = tgt_classes.index('car_body')
    
    for front_back in ['back', 'front']:
        # tgt_mask[mask == src_classes.index(f'{front_back}_bumper')] = tgt_classes.index('car_body')
        tgt_mask[mask == src_classes.index(f'{front_back}_license_plate')] = tgt_classes.index('licence_plate')
        tgt_mask[mask == src_classes.index(f'{front_back}_windshield')] = tgt_classes.index('windshield')

        for left_right in ['left', 'right']:
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_door')] = tgt_classes.index('door')
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_wheel')] = tgt_classes.index('wheel')
            tgt_mask[mask == src_classes.index(f'{front_back}_{left_right}_window')] = tgt_classes.index('window')

    # tgt_mask[mask == src_classes.index('hood')] = tgt_classes.index('car_body')
    # tgt_mask[mask == src_classes.index('roof')] = tgt_classes.index('car_body')
    # tgt_mask[mask == src_classes.index('trunck')] = tgt_classes.index('car_body')

    for left_right in ['left', 'right']:
        # tgt_mask[mask == src_classes.index(f'{left_right}_frame')] = tgt_classes.index('car_body')
        tgt_mask[mask == src_classes.index(f'{left_right}_head_light')] = tgt_classes.index('head_light')
        tgt_mask[mask == src_classes.index(f'{left_right}_mirror')] = tgt_classes.index('mirror')
        tgt_mask[mask == src_classes.index(f'{left_right}_quarter_window')] = tgt_classes.index('window')
        tgt_mask[mask == src_classes.index(f'{left_right}_tail_light')] = tgt_classes.index('tail_light')
    
    return tgt_mask

def cgpart_aeroplane_simplify_v0(mask):
    src_classes = CLASSES['CGPart_aeroplane']
    tgt_classes = CLASSES['CGPart_aeroplane_simplified_v0']
    if isinstance(mask, np.ndarray):
        tgt_mask = np.zeros_like(mask)
    elif isinstance(mask, torch.Tensor):
        tgt_mask = torch.zeros_like(mask)
    tgt_mask[mask != src_classes.index('background')] = tgt_classes.index('body')
    tgt_mask[mask == src_classes.index('fin')] = tgt_classes.index('fin')
    
    for left_right in ['left', 'right']:
        tgt_mask[mask == src_classes.index(f'wing_{left_right}')] = tgt_classes.index('wing')
        tgt_mask[mask == src_classes.index(f'tailplane_{left_right}')] = tgt_classes.index('tailplane')
        tgt_mask[mask == src_classes.index(f'engine_{left_right}')] = tgt_classes.index('engine')

    tgt_mask[mask == src_classes.index('wheel_front')] = tgt_classes.index('wheel')
    tgt_mask[mask == src_classes.index('wheel_back_left')] = tgt_classes.index('wheel')
    tgt_mask[mask == src_classes.index('wheel_back_right')] = tgt_classes.index('wheel')
    
    return tgt_mask

def trans_mask(mask, src_classes, tgt_classes):
    if src_classes == 'dg_car_20' and tgt_classes == 'dg_cat_12':
        return trans_car_mask_20to12(mask)
    if src_classes == 'dg_face' and tgt_classes == 'celebA_8':
        return trans_face_mask_34to8(mask)
    if src_classes == 'celebA_19' and tgt_classes == 'celebA_8':
        return trans_face_mask_19to8(mask)
    if src_classes == 'dg_face' and tgt_classes == 'celebA_19':
        raise NotImplementedError(f'Unsupported transfer for src = {src_classes}, tgt = {tgt_classes}')
    if src_classes == 'CGPart_car' and tgt_classes == 'CGPart_simplified_v0':
        return cgpart_car_simplify_v0(mask)
    if src_classes == 'CGPart_car' and tgt_classes == 'CGPart_simplified_v1':
        return cgpart_car_simplify_v1(mask)
    
    raise ValueError(f'Invalid transfer for src = {src_classes}, tgt = {tgt_classes}')

