import torch
import torch.nn.functional as F
from functools import partial

#----------------------------------------------------------------------------

class TransformPipe(torch.nn.Module):
    def __init__(self,
        wide_crop       = False, # Whether to crop the image/segmentation to 4:3 aspect ratio, especially for Car-20
        resize          = None,  # The target
    ):
        super().__init__()
        self.transform_functions = []

        if wide_crop:
            self.transform_functions.append(wide_crop_fn)
        if resize is not None:
            self.transform_functions.append(partial(resize_fn, size=resize))

    def forward(self, x, y):
        for transform_fn in self.transform_functions:
            x, y = transform_fn(x, y)
        return x, y

#----------------------------------------------------------------------------

def wide_crop_fn(x, y):
    height, width = x.shape[-2:]
    if not height == width:
        assert int(width * 3 // 4) == height, \
            f"wide crop can only be operated on square image, but got {height} x {width}."
        return x, y
    ch = int(height * 3 // 4)

    # Crop x, and crop y if necessary
    x = x[..., (height - ch) // 2: (height + ch) // 2, :]
    if not (y.dim() == 4 or y.dim() == 5):
        return x, y
    assert height == y.shape[-2] and width == y.shape[-1], \
        f"Inconsistent shape between image ({height} x {width}) and label ({y.shape[2]} x {y.shape[3]})"
    y = y[..., (height - ch) // 2: (height + ch) // 2, :]
    return x, y

#----------------------------------------------------------------------------

def resize_fn(x, y, size=None):
    if size is None:
        return x, y
    if x.size(2) == size[0] and x.size(3) == size[1]:
        return x, y

    x = F.interpolate(x, size=size)
    if y.dim() == 4:
        y = F.interpolate(y, size=size, mode='nearest') # Nearest mode to adapt to gt label
    return x, y

#----------------------------------------------------------------------------
