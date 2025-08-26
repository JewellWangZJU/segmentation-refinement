# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from model.gcn_lib import Grapher, act_layer


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


@register_model
def pvig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,18,2] # number of basic blocks in the backbone
            self.channels = [128, 256, 512, 1024] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model
# 添加LWdecoder类
class LWdecoder(nn.Module):
    def __init__(self,
                 in_channels,  # 这是一个列表 [80, 160, 400, 640]
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(LWdecoder, self).__init__()

        self._in_channels = in_channels[:]
        
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        
        self.blocks = nn.ModuleList()
        
        # 确保in_channels是一个列表，并且长度与in_feat_output_strides匹配
        if not isinstance(in_channels, list):
            in_channels = [in_channels] * len(in_feat_output_strides)
        
        if len(in_channels) != len(in_feat_output_strides):
            raise ValueError("in_channels and in_feat_output_strides must have the same length")
        
        # 为每个特征级别创建适配层
        self.adapter_layers = nn.ModuleList()
        for in_channel in in_channels:
            # 添加一个适配层来调整通道数
            adapter = nn.Sequential(
                nn.Conv2d(in_channel, out_channels, 1, bias=False),
                norm_fn(**norm_fn_args),
                nn.ReLU(inplace=True)
            )
            self.adapter_layers.append(adapter)
        
        # 创建上采样块
        for i, in_feat_os in enumerate(in_feat_output_strides):
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))
            num_layers = num_upsample if num_upsample > 0 else 1
            
            layers = []
            for idx in range(num_layers):
                # 所有层都使用out_channels，因为我们已经通过适配层调整了通道数
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                        norm_fn(**norm_fn_args),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2) if idx < num_layers - 1 else nn.Identity(),
                    )
                )
            
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, feat_list: list):
        inner_feat_list = []
        
        # 首先通过适配层调整所有特征图的通道数
        adapted_feats = []
        assert len(feat_list) == len(self.blocks), f"[DBG] decoder got {len(feat_list)} feats, expected {len(self.blocks)}"
        for idx, feat in enumerate(feat_list):
            # print(f'[DBG] decoder in feat[{idx}]:', feat.shape, f'(expect in_channels={self._in_channels[idx]}')

            if idx < len(self.adapter_layers):
                adapted_feat = self.adapter_layers[idx](feat)
                adapted_feats.append(adapted_feat)
        
        # 然后对每个适配后的特征进行上采样
        for idx, block in enumerate(self.blocks):
            if idx < len(adapted_feats):
                decoder_feat = block(adapted_feats[idx])
                inner_feat_list.append(decoder_feat)
        
        if not inner_feat_list:
            return None
        
        # 将所有特征图上采样到相同的尺寸并求平均
        target_size = inner_feat_list[0].shape[2:]
        upsampled_feats = []
        for feat in inner_feat_list:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled_feats.append(feat)
        
        out_feat = sum(upsampled_feats) / len(upsampled_feats)
        return out_feat

# 添加Vigseg类
class VigSeg(nn.Module):
    def __init__(self, opt):
        super(VigSeg, self).__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=channels[0], act=act)
        
        # 动态计算pos_embed尺寸
        stem_output_size = opt.img_size // 4  # Stem有2次stride=2的下采样
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], stem_output_size, stem_output_size))
        HW = stem_output_size * stem_output_size

        self.encoder = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        idx = 0
        
        for i in range(len(blocks)):
            if i > 0:
                self.downsample_layers.append(Downsample(channels[i-1], channels[i]))
            else:
                self.downsample_layers.append(nn.Identity())
            
            stage = nn.ModuleList()
            for j in range(blocks[i]):
                stage.append(nn.Sequential(
                    Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                            bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                            relative_pos=True),
                    FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                ))
                idx += 1
            self.encoder.append(stage)
            if i > 0:
                HW = HW // 4  # 每次下采样后特征图尺寸减半

        # Decoder
        self.decoder = LWdecoder(
            in_channels=channels,
            out_channels=opt.decoder_out_channels,
            in_feat_output_strides=[4, 8, 16, 32],
            out_feat_output_stride=4,
            norm_fn=nn.BatchNorm2d
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(opt.decoder_out_channels, opt.decoder_out_channels//2, 3, padding=1),
            nn.BatchNorm2d(opt.decoder_out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(opt.decoder_out_channels//2, opt.n_classes, 1)
        )

    def forward(self, x):
        features = []
        stem_output = self.stem(x)
        # print('[DBG] stem:', stem_output.shape)
        
        # 确保pos_embed尺寸匹配
        B, C, H, W = stem_output.shape
        if self.pos_embed.shape[2:] != (H, W):
            pos_embed_resized = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=True)
        else:
            pos_embed_resized = self.pos_embed
        # print('[DBG] pos_embed_resized:', pos_embed_resized.shape)
        
        x = stem_output + pos_embed_resized
        
        for i in range(len(self.encoder)):
            if i > 0:
                x = self.downsample_layers[i](x)
                # print(f'[DBG] downsample to stage {i}:', x.shape)
            for block in self.encoder[i]:
                x = block(x)
            features.append(x)
            # print(f"[DBG] stage{i}_out:", x.shape)

        
        # assert只有4个尺度
        assert len(features) == 4, f"[DBG] feature len= {len(features)} != 4"

        x = self.decoder(features)
        x = self.seg_head(x)
        return x

# 添加模型注册函数
@register_model
def vig_seg_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]
            self.channels = [80, 160, 400, 640]
            self.n_classes = num_classes
            self.emb_dims = 1024
            self.img_size = kwargs.get('img_size', 224) # 从kwargs获取img_size,默认为224
            self.decoder_out_channels = 256

    opt = OptInit(**kwargs)
    model = VigSeg(opt)
    return model


