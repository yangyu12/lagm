import torch
import time
import numpy as np
from external.stylegan2 import dnnlib

#----------------------------------------------------------------------------

def calc_metrics(
    model,
    testloader,
    device,
    nC          = None,
):
    start_time = time.time()
    if nC is None:
        nC = model.output_channels
    C = np.zeros((nC, nC), dtype=np.int64)  # Confusion matrix: [Pred x GT]

    model = model.to(device)
    for data in testloader:
        img, gt_mask = data[:2]

        # validate on student
        img = img.to(device, dtype=torch.float32)
        gt_mask = gt_mask.to(device, dtype=torch.long)
        pred_logits = model(img)
        pred_mask = pred_logits.max(dim=1)[1]
        pred_mask = pred_mask.cpu().numpy().reshape(-1)
        gt_mask = gt_mask.cpu().numpy().reshape(-1)
        # Note: gt_mask may contain ignorance pixels
        C += np.bincount(
            nC * pred_mask[gt_mask < nC] + gt_mask[gt_mask < nC],  # the value is in the range of [0, nC**2)
            minlength=nC ** 2).reshape(nC, nC)  # reshape to [Pred x GT]

    results = {}
    C = C.astype(np.float)
    results['pACC'] = C.diagonal().sum() / C.sum()
    # IoU = tp / (tp + fn + fp)
    union = C.sum(axis=1) + C.sum(axis=0) - C.diagonal()
    union[union == 0] = 1e-8
    iou_vals = C.diagonal() / union  # (nC,)
    results['mIoU'] = iou_vals.mean()
    results['fg_mIoU'] = iou_vals[1:].mean()

    total_time = time.time() - start_time

    return dict(
        results=results,
        metrics=['pACC', 'mIoU', 'fg_mIoU'],
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
    )

#----------------------------------------------------------------------------
