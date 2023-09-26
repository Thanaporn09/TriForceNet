import math
import warnings
from functools import partial
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init,NORM_LAYERS,build_conv_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert


class MixFFN(BaseModule):
    """An implementation of MixFFN of PVT.
    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, activate, drop, fc2, drop]
        if use_conv:
            layers.insert(1, dw_conv)
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

#@NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

 
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv2 = build_conv_layer(
            conv_cfg,
            in_channels, 
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.conv4 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = self.conv4(x)

            out = self.conv2(x)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Conv_Block(BaseModule):
    """Conv_Block.
    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
    Note:
        There are two equivalent implementations:
        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back
        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 mlp_ratio=2.,
                 linear_pw_conv=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        mid_channels = int(mlp_ratio * in_channels)
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels)
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            )
        self.linear_pw_conv = linear_pw_conv
        self.norm1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.norm2 = build_norm_layer(norm_cfg, out_channels)[1]

        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(mid_channels, out_channels)
        self.act = build_activation_layer(act_cfg)
        self.act2 = build_activation_layer(act_cfg)
        self.act3 = build_activation_layer(act_cfg)
       
        self.drop_path1 = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)) if drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x):
       
        x1 = self.depthwise_conv(x)
        x1 = self.norm1(x1)
        x1 = self.act3(x1)
        if self.linear_pw_conv:
            x1 = x1.permute(0,2,3,1)
        x1 = self.pointwise_conv1(x1)
        x1 = self.act(x1)
        if self.linear_pw_conv:
            x1 = x1.permute(0,3,1,2)
        x1 = self.drop_path1(x1)

        x2 = self.conv3(x)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
       
        out = x1+x2
        return out

class Bilinear_pooling(MultiheadAttention):
    """
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 norm_cfg=dict(type='LN'),
                 chratio=1,
                 spratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.c_mid = embed_dims//chratio
        self.scale_factor = int(np.log2(spratio))
        down_k = []
        for i in range(self.scale_factor):
            self.avgpool = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )
            down_k.append(self.avgpool)
        self.down_xk = nn.Sequential(*down_k)

        self.convK = Conv2d(
            in_channels=embed_dims,
            out_channels=self.c_mid,
            kernel_size=1,
            groups=self.c_mid
        )

        down_v = []
        for j in range(self.scale_factor):
            self.dwconv = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )

            down_v.append(self.dwconv)
        self.down_xv = nn.Sequential(*down_v)

        self.convV = Conv2d(
            in_channels=embed_dims,
            out_channels=self.c_mid,
            kernel_size=1,
            groups=self.c_mid
        )
        self.convQ = Conv2d(
            in_channels=embed_dims,
            out_channels=self.c_mid,
            kernel_size=1,
            groups=self.c_mid
        )
        self.normq = build_norm_layer(norm_cfg, self.c_mid)[1]
        self.normk = build_norm_layer(norm_cfg, self.c_mid)[1]
        self.reconv = Conv2d(self.c_mid,embed_dims,kernel_size=1)

    def forward(self, x, hw_shape, identity=None):
        n,l,c = x.shape
        x = nlc_to_nchw(x, hw_shape)

        x_k = self.convK(self.down_xk(x))
        x_v = self.convV(self.down_xv(x))
        x_q = self.convQ(x)

        x_q = nchw_to_nlc(x_q)
        x_q = self.normq(x_q)
        x_q = F.softmax(x_q,dim=-1)

        x_k = nchw_to_nlc(x_k)
        x_k = self.normk(x_k).permute(0,2,1)
        x_k = F.softmax(x_k,dim=-1)

        x_v = nchw_to_nlc(x_v)
        
        x_kv = torch.bmm(x_k,x_v)
        out = x_q.matmul(x_kv)
        out = nlc_to_nchw(out, hw_shape)
        
        out = self.reconv(out)
        if identity is None:
            identity = x

        output = identity + self.dropout_layer(self.proj_drop(out))
        output = nchw_to_nlc(output)
        return output

class NewTF_EncoderLayer(BaseModule):
    """Implements one encoder layer in PVT.
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 chratio=1,
                 spratio=1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 use_conv_ffn=False,
                 init_cfg=None):
        super(NewTF_EncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = Bilinear_pooling(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            chratio=chratio,
            spratio=spratio,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            norm_cfg=norm_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=None)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)

        return x
   
@BACKBONES.register_module()
class SeqHTC(BaseModule):
    """
    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[3, 3],
                 strides=[2, 2],
                 paddings=[1, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[8, 8],
                 chratio = [2, 2],
                 spratio = [2, 2],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_after_stage=False,
                 use_conv_ffn=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 convert_weights=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.chratio = chratio
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.spratio = spratio

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.preconv_block= Conv_Block(
                                in_channels,
                                embed_dims,
                                drop_path_rate=dpr[0]
                                )
        cur_conv = 0
        self.con_stage = ModuleList()
        for num_c in range(int(num_stages//2)): 
            in_channels = embed_dims*num_heads[2*num_c]
            out_dims = embed_dims*num_heads[(2*num_c)+1]
            down_layer = nn.Conv2d(
                            in_channels,
                            out_dims,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            dilation=1
                        )
            conv_layers = ModuleList()
            for j in range(num_layers[2*num_c]):
                conv_layers.extend([
                    Conv_Block(
                        out_dims,
                        out_dims,
                        drop_path_rate=dpr[cur_conv+j]
                    )
                ])
            self.con_stage.append(ModuleList([down_layer,conv_layers]))
            cur_conv += num_layers[2*num_c]

        self.layers = ModuleList()
        for num_t in range(int(num_stages//2)):
            num_layer = num_layers[(2*num_t)]
            in_channelt = embed_dims * num_heads[(num_t)]
            embed_dims_i = embed_dims * num_heads[(2*num_t)]
            patch_embed = PatchEmbed(
                in_channels=in_channelt,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[num_t],
                stride=strides[num_t],
                padding=paddings[num_t],
                bias=True,
                norm_cfg=norm_cfg)

            layers = ModuleList()
            layers.extend([
                NewTF_EncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[num_t],
                    feedforward_channels=mlp_ratios[num_t] * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    chratio=self.chratio[num_t],
                    spratio=self.spratio[num_t],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    use_conv_ffn=use_conv_ffn) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            if norm_after_stage:
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            else:
                norm = nn.Identity()
            self.layers.append(ModuleList([patch_embed, layers, norm]))
            cur += num_layer

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if self.convert_weights:
                # Because pvt backbones are not supported by mmcls,
                # so we need to convert pre-trained weights to match this
                # implementation.
                state_dict = pvt_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []
        x = self.preconv_block(x)
        num_layer = 0
        for i, layer in enumerate(self.layers):
            ### transformer layers ###
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if num_layer in self.out_indices:
                outs.append(x)
            if i > 0:
                num_layer += 1
            ### conv_layer ###
            x = self.con_stage[i][0](x)
            for conv_block in self.con_stage[i][1]:
                x = conv_block(x)
            num_layer += 1
            if num_layer in self.out_indices:
                outs.append(x)
                #print(num_layer)
        return outs
