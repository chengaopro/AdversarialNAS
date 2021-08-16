# @Date    : 2019-10-22
# @Author  : Chen Gao

from torch import nn
import torch.nn.functional as F
import pdb

UP_MODES = ['nearest', 'bilinear']
NORMS = ['in', 'bn']

# 7
PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# 5
PRIMITIVES_wo_act = [
  'conv_1x1',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

# 3
PRIMITIVES_up = [
    'nearest',
    'bilinear',
    'ConvTranspose'
]

# 6
PRIMITIVES_down = [
    'avg_pool',
    'max_pool',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


OPS = {
    'none': lambda in_ch, out_ch, stride, sn, act: Zero(),
    'skip_connect': lambda in_ch, out_ch, stride, sn, act: Identity(),
    'conv_1x1': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 1, stride, 0, sn, act),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

OPS_down = {
    'avg_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Avg'),
    'max_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Max'),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

UPS = {
    'nearest': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='nearest'),
    'bilinear': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='bilinear'),
    'ConvTranspose': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='convT')
}


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, sn, act):
        super(Conv, self).__init__()
        if sn:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        if act:
            self.op = nn.Sequential(nn.ReLU(), self.conv)
        else:
            self.op = nn.Sequential(self.conv)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, sn, act):
        super(DilConv, self).__init__()
        if sn:
            self.dilconv = nn.utils.spectral_norm(
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            self.dilconv = \
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        if act:
            self.op = nn.Sequential(nn.ReLU(), self.dilconv)
        else:
            self.op = nn.Sequential(self.dilconv)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Up, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'convT':
            self.convT = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_ch,
                                   bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )
        else:
            self.c = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )

    def forward(self, x):
        if self.up_mode == 'convT':
            return self.convT(x)
        else:
            return self.c(F.interpolate(x, scale_factor=2, mode=self.up_mode))


class Pool(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Pool, self).__init__()
        self.pool_mode = mode

        if self.pool_mode == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif self.pool_mode == 'Max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

    def forward(self, x):
        return self.pool(x)


class MixedOp(nn.Module):
    def __init__(self, in_ch, out_ch, stride, sn, act):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class MixedOp_sn_wo_act(nn.Module):
    def __init__(self, in_ch, out_ch, stride, sn, act):
        super(MixedOp_sn_wo_act, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES_wo_act:
            op = OPS[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class MixedDown(nn.Module):
    def __init__(self, in_ch, out_ch, stride, sn, act):
        super(MixedDown, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES_down:
            op = OPS_down[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class MixedUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MixedUp, self).__init__()
        self.ups = nn.ModuleList()
        for primitive in PRIMITIVES_up:
            up = UPS[primitive](in_ch, out_ch)
            self.ups.append(up)

    def forward(self, x, weights):
        return sum(w * up(x) for w, up in zip(weights, self.ups))


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, num_skip_in=0, norm=None):
        super(Cell, self).__init__()

        self.up0 = MixedUp(in_channels, out_channels)
        self.up1 = MixedUp(in_channels, out_channels)

        self.c0 = MixedOp(out_channels, out_channels, 1, False, True)
        self.c1 = MixedOp(out_channels, out_channels, 1, False, True)
        self.c2 = MixedOp(out_channels, out_channels, 1, False, True)
        self.c3 = MixedOp(out_channels, out_channels, 1, False, True)
        self.c4 = MixedOp(out_channels, out_channels, 1, False, True)

        self.up_mode = up_mode
        self.norm = norm

        # no norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, weights_normal, weights_up, skip_ft=None):

        node0 = self.up0(x, weights_up[0])
        node1 = self.up1(x, weights_up[1])
        _, _, ht, wt = node0.size()

        node2 = self.c0(node0, weights_normal[0]) + self.c1(node1, weights_normal[1])

        h_skip_out = node2

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                node2 += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))

        node3 = self.c2(node0, weights_normal[2]) + self.c3(node1, weights_normal[3]) + self.c4(node2, weights_normal[4])

        return h_skip_out, node3


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, activation=nn.ReLU(), downsample=False):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample

        self.c0 = MixedOp_sn_wo_act(in_channels, out_channels, 1, True, False)
        self.c1 = MixedOp_sn_wo_act(in_channels, out_channels, 1, True, False)
        self.c2 = MixedOp_sn_wo_act(in_channels, out_channels, 1, True, False)
        self.c3 = MixedOp(out_channels, out_channels, 1, True, True)
        self.c4 = MixedOp(out_channels, out_channels, 1, True, True)

        if self.downsample:
            self.down0 = MixedDown(out_channels, out_channels, 2, True, True)
            self.down1 = MixedDown(out_channels, out_channels, 2, True, True)
        else:
            self.c5 = MixedOp(in_channels, out_channels, 1, True, True)
            self.c6 = MixedOp(in_channels, out_channels, 1, True, True)

    def forward(self, x, weights_normal=None, weights_down=None):
        node0 = self.c0(x, weights_normal[0])
        node1 = self.c1(x, weights_normal[1]) + self.c3(node0, weights_normal[3])
        node2 = self.c2(x, weights_normal[2]) + self.c4(node0, weights_normal[4])

        if self.downsample:
            node3 = self.down0(node1, weights_down[0]) + self.down1(node2, weights_down[1])
        else:
            return

        return node3


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample

        self.c0 = MixedOp(in_channels, out_channels, 1, True, True)
        self.c1 = MixedOp(in_channels, out_channels, 1, True, True)
        self.c2 = MixedOp(in_channels, out_channels, 1, True, True)
        self.c3 = MixedOp(out_channels, out_channels, 1, True, True)
        self.c4 = MixedOp(out_channels, out_channels, 1, True, True)

        if self.downsample:
            self.down0 = MixedDown(out_channels, out_channels, 2, True, True)
            self.down1 = MixedDown(out_channels, out_channels, 2, True, True)
        else:
            self.c5 = MixedOp(in_channels, out_channels, 1, True, True)
            self.c6 = MixedOp(in_channels, out_channels, 1, True, True)

    def forward(self, x, weights_normal=None, weights_down=None):

        node0 = self.c0(x, weights_normal[0])
        node1 = self.c1(x, weights_normal[1]) + self.c3(node0, weights_normal[3])
        node2 = self.c2(x, weights_normal[2]) + self.c4(node0, weights_normal[4])

        if self.downsample:
            node3 = self.down0(node1, weights_down[0]) + self.down1(node2, weights_down[1])
        else:
            return

        return node3