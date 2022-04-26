import torch
import torch.nn.functional as F
from torch_utils import training_stats

#----------------------------------------------------------------------------

class GMLoss:
    def __init__(self, device,
        G, A, S, T, matcher, params_for_gm,
    ):
        self.device = device
        self.G = G
        self.A = A
        self.S = S
        self.T = T
        self.matcher = matcher
        self.params_for_gm = list(params_for_gm)

    def synthesize_data(self, z, c):
        # Generate image & features
        img, features = self.G(z, c)

        # Label the synthetic image with annotator
        auto_label = self.A(features)

        # Post-process the synthetic image and mask
        if self.T is not None:
            img, auto_label = self.T(img, auto_label)
        return img, auto_label

    def forward(self, phase, lbl_img, manual_label, syn_z, syn_c, syn_loss_weight=1.0, real_loss_weight=0.0):
        assert phase in ['A', 'S']
        do_Amain = (phase in ['A'])
        do_Smain = (phase in ['S'])

        if do_Amain:
            # Compute loss & bp gradients on labeled minibatch (target for gradient matching)
            pred_lbl = self.S(lbl_img)
            loss_lbl = F.cross_entropy(pred_lbl, manual_label)
            training_stats.report('Loss/GM/loss_label', loss_lbl)
            param_grad_lbl = torch.autograd.grad(loss_lbl, self.params_for_gm)
            param_grad_lbl = list((_.detach().clone() for _ in param_grad_lbl))

            # Compute loss & bp gradients on synthetic minibatch
            syn_img, auto_label = self.synthesize_data(syn_z, syn_c)
            auto_label = F.softmax(auto_label, dim=1)
            pred_syn = self.S(syn_img)
            loss_syn = torch.sum(- auto_label * F.log_softmax(pred_syn, dim=1), dim=1).mean()
            training_stats.report('Loss/GM/loss_syn', loss_syn)
            param_grad_syn = torch.autograd.grad(loss_syn, self.params_for_gm, create_graph=True)

            # Compute gradient matching loss & bp gradients to A
            loss_gm = self.matcher.match(param_grad_syn, param_grad_lbl)
            training_stats.report('Loss/GM/loss_gm', loss_gm)
            loss_gm.backward()

        if do_Smain:
            # Compute loss & bp gradients on synthetic minibatch
            syn_img, auto_label = self.synthesize_data(syn_z, syn_c)
            auto_label = F.softmax(auto_label, dim=1)
            pred_syn = self.S(syn_img)
            loss_syn = torch.sum(- auto_label * F.log_softmax(pred_syn, dim=1), dim=1).mean()
            training_stats.report('Loss/S/loss_syn', loss_syn)

            training_stats.report('Loss/S/loss', loss_syn)
            loss_syn.backward()

#----------------------------------------------------------------------------

class GradientMatcher:
    def __init__(self,
        metric          = 'cosine', # 'cosine', 'l2'
        reduction       = 'mean',   # 'mean' or 'sum'
    ):
        self.metric = metric
        self.reduction = reduction

    def match(self, grad_syn, grad_real):
        return gradient_matching(grad_syn, grad_real, metric=self.metric, reduction=self.reduction)

#----------------------------------------------------------------------------

def gradient_matching(param_grad_syn, param_grad_lbl, metric='cosine', reduction='sum'):
    dis = 0
    assert len(param_grad_syn) == len(param_grad_lbl), f'{len(param_grad_syn)}, {len(param_grad_lbl)}'

    for x, y in zip(param_grad_syn, param_grad_lbl):
        dis = dis + distance_wb(x, y, metric, reduction=reduction)
    if reduction == 'mean':
        num_params = len(param_grad_syn)
        dis = dis / num_params
    return dis

def distance_wb(x, y, metric, reduction='sum'):
    shape = x.shape
    if len(shape) == 4:         # weight of conv layer (out, in, kh, kw)
        x = x.reshape(shape[0], shape[1] * shape[2] * shape[3])
        y = y.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:       # layernorm (C, h, w) TODO: ???
        x = x.reshape(shape[0], shape[1] * shape[2])
        y = y.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:       # weight of linear layer, (out, in)
        tmp = 'do nothing'      # do not need to reshape
    elif len(shape) == 1:       # BN/IN:  (C, ); GN: x; Bias of a lot layers
        x = x.reshape(1, shape[0])
        y = y.reshape(1, shape[0])
        return 0

    if metric == 'cosine':
        distance = torch.sum(1. - F.cosine_similarity(x, y, dim=1, eps=1e-6))
    elif metric == 'l2':
        distance = torch.sum((x - y) ** 2, dim=1).sqrt().sum()
    elif metric == 'dotprod': # dot product
        distance = - torch.sum(x * y)
    else:
        raise NotImplementedError

    if reduction == 'mean':
        num_groups = x.shape[0]
        distance = distance / num_groups

    return distance

#----------------------------------------------------------------------------


