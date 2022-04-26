import re
import torch
import numpy as np
import torch.nn.functional as F
from torch_utils import persistence
from collections import OrderedDict
import dnnlib
import legacy
import math
from functools import partial
import torchvision
from training import stylegan1

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynDataGenerator(torch.nn.Module):
    """Suitable for StyleGAN2"""
    def __init__(self,
        generator_pkl,                                   #
        regex               = r'.*synthesis\.b(\d+)$',   #
        truncation_psi      = 0.7,
        truncation_cutoff   = 8,
    ):
        super().__init__()

        with dnnlib.util.open_url(generator_pkl) as f:
            print(generator_pkl)
            self.G = legacy.load_network_pkl(f)['G_ema']

        # Default
        # ---
        # b4    conv1           (512, 4, 4)
        # b8    conv0/conv1     (512, 8, 8)
        # b16   conv0/conv1     (512, 16, 16)
        # b32   conv0/conv1     (512, 32, 32)
        # b64   conv0/conv1     (512, 64, 64)
        # b128  conv0/conv1     (256, 128, 128)
        # b256  conv0/conv1     (128, 256, 256)
        # b512  conv0/conv1     (64, 512, 512)
        # -------------------------------------
        # f_dim = 512 * (1 + 2 + 2 + 2 + 2) + 256 * 2 + 128 * 2 + 64 * 2 = 5504

        # Copy some attributes for convenience
        self.z_dim = self.G.z_dim
        self.c_dim = self.G.c_dim
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        # extract intermediate features with hooks.
        # Reference: https://medium.com/the-owl/using-forward-hooks-to-extract-intermediate-layer-outputs-from-a-pre-trained-model-in-pytorch-1ec17af78712
        self.inner_features = []
        self.fhooks = []
        self.f_shape = None
        for name, mod in self.G.named_modules():
            if re.fullmatch(regex, name):
                self.fhooks.append(mod.register_forward_hook(self.forward_hook))

    def forward_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.inner_features.append(output.float())
        else: # StyleGAN2 synthesis block typically return 2 tensors
            assert len(output) > 1
            channels = [x.size(1) for x in output]
            max_ch_idx = channels.index(max(channels))
            self.inner_features.append(output[max_ch_idx].float())

    def forward(self, z, c, noise_mode='random'):
        self.inner_features = []
        out = self.G(z, c, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff, noise_mode=noise_mode)
        
        if self.f_shape is None:
            self.f_shape = [x.shape[1:] for x in self.inner_features]
        return out, self.inner_features

#----------------------------------------------------------------------------

@persistence.persistent_class
class StyleGANGenerator(torch.nn.Module):
    """Ver1"""
    def __init__(self,
        generator_pt,  #
        img_resolution,
        w_avg_npy,
        img_channels        = 3,
        truncation_psi      = 0.7,
        truncation_cutoff   = 8,
    ):
        super().__init__()
        self.z_dim = 512
        self.c_dim = 0
        self.w_dim = 512
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = stylegan1.G_synthesis(resolution=img_resolution)
        self.mapping = stylegan1.G_mapping()
        w_avg = torch.from_numpy(np.load(w_avg_npy)).to(dtype=torch.float32)
        self.truncation = stylegan1.Truncation(w_avg, max_layer=truncation_cutoff, truncation_psi=truncation_psi)
        # Configure max_layer: 8 for car & face; 7 for cat

        # Load pre-trained generator
        self.load_pt_file(generator_pt)
        self.f_shape = None

    def load_pt_file(self, pt_file):
        # Pre-check the shape
        model_state_dict = torch.nn.Sequential(
            OrderedDict([('g_mapping', self.mapping), ('truncation', self.truncation), ('g_synthesis', self.synthesis)])
        ).state_dict()
        state_dict = torch.load(pt_file)
        mismatch_keys = []
        for k in state_dict.keys():
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    print(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint, shape_model)
                    )
                    mismatch_keys.append(k)
        for k in mismatch_keys:
            state_dict.pop(k)

        incompatible = torch.nn.Sequential(
            OrderedDict([('g_mapping', self.mapping), ('truncation', self.truncation), ('g_synthesis', self.synthesis)])
        ).load_state_dict(torch.load(pt_file), strict=False)
        if incompatible.missing_keys:
            msg = "Some model parameters are not in the checkpoint:\n"
            msg += "\n".join(incompatible.missing_keys)
            print(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint contains parameters not used by the model:\n"
            msg += "\n".join(incompatible.unexpected_keys)
            print(msg)

    def forward(self, z, c, **kwargs):
        ws = self.mapping(z)
        ws = self.truncation(ws)
        return self.forward_with_w(ws)

    def forward_with_w(self, ws):
        # Broadcast w
        if ws.ndim == 2:
            ws = ws.unsqueeze(1).expand(-1, 18, -1)
        img, inner_features = self.synthesis(ws)

        # Hack for Face (1024 x 1024)
        if self.img_resolution == 1024:
            img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
            inner_features[-1] = F.interpolate(inner_features[-1], size=(512, 512), mode='bilinear', align_corners=False)
            inner_features[-2] = F.interpolate(inner_features[-2], size=(512, 512), mode='bilinear', align_corners=False)

        if self.f_shape is None:
            self.f_shape = [x.shape[1:] for x in inner_features]
        return img, inner_features

#----------------------------------------------------------------------------

@persistence.persistent_class
class FPNLabelBranch(torch.nn.Module):
    def __init__(self,
        input_shapes,            # The shapes of the multi-resolution feature maps
        tmp_channels,              # The number of channels of the propagated feature map
        output_channels,           # The number of classes
        num_layers      = 3,       # The number of 3x3 conv layers used to predict pixel labels
        fuse_bn_relu    = False,
        use_factor      = False,
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)
        # added 
        self.output_channels = output_channels

        # Lateral layer producing feature map to be propagated
        for i in range(self.num_features):
            channels = self.input_channels[i]
            conv = torch.nn.Conv2d(channels, tmp_channels, kernel_size=1)
            setattr(self, f'lateral{i}', conv)
        self.fuse_factor = 1. / math.sqrt(self.num_features) if use_factor else 1.

        # Segmentation head
        head = [torch.nn.BatchNorm2d(tmp_channels), torch.nn.ReLU(inplace=True)] if fuse_bn_relu else []
        for i in range(num_layers):
            last_layer = (i == num_layers - 1)
            if last_layer:  # last layer
                head.append(torch.nn.Conv2d(tmp_channels, output_channels, kernel_size=3, padding=1, bias=True))
                break
            else:
                head.append(torch.nn.Conv2d(tmp_channels, tmp_channels, kernel_size=3, padding=1, bias=False))
            head.append(torch.nn.BatchNorm2d(tmp_channels))
            head.append(torch.nn.ReLU(inplace=True))
        self.to_logit = torch.nn.Sequential(*head)

    def forward(self, x):
        # FPN
        all_features = x
        x = None
        for i in range(self.num_features):
            assert all_features[i].size(1) == self.input_channels[i]
            lateral = getattr(self, f'lateral{i}')
            y = self.fuse_factor * lateral(all_features[i])
            if x is None:
                x = y
            else:
                if x.size(2) != y.size(2) or x.size(3) != y.size(3):
                    x = torch.nn.functional.interpolate(x, size=(y.size(2), y.size(3)),
                                                        mode='bilinear', align_corners=False)
                x = x.add_(y)

        # Segmentation head
        return self.to_logit(x)

#----------------------------------------------------------------------------

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        up              = False,
        down            = False,
        bilinear        = True,
    ):
        super().__init__()
        self.up_scale = None
        self.down_scale = None
        assert not (up and down)
        if up:
            self.up_scale = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                            torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        if down:
            self.down_scale = torch.nn.MaxPool2d(2)

        which_conv2d = partial(torch.nn.Conv2d, kernel_size=3, padding=1)

        # self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = torch.nn.BatchNorm2d(out_channels)
        # self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = which_conv2d(in_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = which_conv2d(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2=None):
        x = x1

        if self.up_scale is not None:
            assert x2 is not None
            x1 = self.up_scale(x1)
            # pad x1 if the size does not match the size of x2
            dh = x2.size(2) - x1.size(2)
            dw = x2.size(3) - x1.size(3)
            x1 = torch.nn.functional.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x2, x1], dim=1)

        if self.down_scale is not None:
            x = self.down_scale(x1)

        x = torch.nn.functional.relu_(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu_(self.bn2(self.conv2(x)))
        return x

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        input_channels  = 3,
        output_channels = 2,
        bilinear        = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.inc = UNetBlock(self.input_channels, 64)

        self.down1 = UNetBlock(64, 128, down=True)
        self.down2 = UNetBlock(128, 256, down=True)
        self.down3 = UNetBlock(256, 512, down=True)
        self.down4 = UNetBlock(512, 512, down=True)

        self.up1 = UNetBlock(1024, 256, up=True, bilinear=bilinear)
        self.up2 = UNetBlock(512, 128, up=True, bilinear=bilinear)
        self.up3 = UNetBlock(256, 64, up=True, bilinear=bilinear)
        self.up4 = UNetBlock(128, 64, up=True, bilinear=bilinear)

        self.outc = torch.nn.Conv2d(64, self.output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

#----------------------------------------------------------------------------

@persistence.persistent_class
class DeepLabv3(torch.nn.Module):
    def __init__(self,
        input_channels     = 3,
        output_channels    = 2,
        pretrain           = True
    ):
        super().__init__()
        self.output_channels = output_channels

        self.main_net = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=False, progress=False, num_classes=output_channels, aux_loss=None,
            pretrained_backbone=pretrain
        )

        input_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        input_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('input_mean', input_mean)
        self.register_buffer('input_std', input_std)

    def forward(self, x):
        # Assume x is in range [-1, 1]
        x = 0.5 * (x + 1)
        x = (x - self.input_mean) / self.input_std

        out = self.main_net(x)
        return out['out']

#----------------------------------------------------------------------------
