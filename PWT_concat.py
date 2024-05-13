from copy import deepcopy
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import einsum
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init, kaiming_init)
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, build_dropout, PatchEmbed, PatchMerging
from mmengine.registry import MODELS

# 大致介绍
# 本代码为将原图像分割为多个子图像后的代码，非将原图像进行映射后的代码
# 流程：映射->attentnion->串行链接/拼接链接->总的映射层

# DSA = WSA + PSA 
# 注意：这里的attention操作没有对结果进行映射，映射层放在了最后

class DSA(BaseModule):

    def __init__(self, token, embed_dims, num_heads = 1, win_size = 1, attn_drop=0., proj_drop=0., qkv_bias=True, init_cfg=None):
        super(DSA, self).__init__(init_cfg) # 父类构造函数的初始化
        
        self.token = int(token), # 图像序列的某一个方向上的长度
        self.num_heads = num_heads # 分组头的数目
        self.head_dim = embed_dims // num_heads # 每个头的通道数目
        
        self.scale = self.head_dim ** -0.5 # 修改
        self.win_size = win_size # 用于表示此时所分化的窗口大小
        
        self.attend = nn.Sequential(
            nn.Softmax(-1),
            nn.Dropout(attn_drop) 
        )
        self.to_qkv = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 3, bias = qkv_bias),
            nn.Dropout(proj_drop)
        )
        
        if self.win_size != 1:
            # 设置窗口的相关操作
            win = token // win_size if token % win_size == 0 else (token // win_size + 1) # 上取整操作 表示单个维度上的窗口数量
            self.m = win * win_size 
            self.up = nn.AdaptiveAvgPool2d((self.m, self.m)) # 用来修正原窗口大小下的图像
            self.down = nn.AdaptiveAvgPool2d((token, token))  # 用来恢复原来的大小的卷积层
            self.sr = nn.AvgPool2d(kernel_size=win, stride=win) # 用来计算相应的令牌 这一步做了消融实验
            # 维度问题 
            self.norm_qk = nn.LayerNorm(embed_dims) 
            # 已修改 self.norm_qk = nn.LayerNorm(embed_dims // 2) 问题
            self.g_qk = nn.Sequential( # 计算对应令牌Q，K 涉及到维度
                nn.Linear(embed_dims, 2 * embed_dims, bias=qkv_bias),
                nn.Dropout(proj_drop)
            )
            
            # 以下这一部分我认为就是用来作为相对位置矩阵 暂不明确
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2*win-1) ** 2,self.num_heads))
            # 维度问题
            
            h = torch.arange(win) 
            w = torch.arange(win) 
            coords = torch.stack(torch.meshgrid([h, w])) 
            coords = torch.flatten(coords, 1)
            coords = coords[:, :, None] - coords[:, None, :]
            coords = coords.permute(1, 2, 0).contiguous()
            coords[:, :, 0] += win-1
            coords[:, :, 1] += win-1
            coords[:, :, 0] *= 2*win - 1
            coords = coords.sum(-1)
            self.register_buffer("relative_index", coords) 
            
    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, H, W, C * 3)
        
        if self.win_size != 1:
            # WSA
            qkv = qkv.permute(0,3,1,2).contiguous()
            qkv = self.up(qkv).permute(0,2,3,1).contiguous()
            H = self.m
            W = self.m
            win = H // self.win_size
            qkv = qkv.reshape(B, H // win, win, W // win, win, 3, self.num_heads, self.head_dim). \
                    permute(5, 0, 1, 3, 6, 2, 4, 7).contiguous().reshape(3, B, H * W // win**2, self.num_heads, win**2, self.head_dim)
            # (B, H // win, win, W // win, win, 3, self.num_heads,self.head_dims) -> (3,B, H // win,W // win, self.num_heads,win,win,self.head_dim) -> (3, B, H * W // win3**2, num_heads,win ** 2,head_dim)
        
            # 位置偏置矩阵       
            bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(win**2, win**2, -1)
            bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
        
            # 计算attention
            atte_l = self.attend((qkv[0] @ qkv[1].transpose(-2, -1).contiguous()) * self.scale + bias)
            xl = (atte_l @ qkv[2]).permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W // win**2, win**2, C). \
                    reshape(B, H // win, W // win, win, win, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, C)
            
            # (B, H * W // win**2, num_heads,win**2, head_dim) -> (B, H * W // win**2, win**2, num_heads,head_dim) -> (B, H * W // win**2, win**2, C) -> (B, H // win, W // win, win, win, C) -> (B,H // win,win,W // win,win,C) -> (B, H, W, C)
            
            # PSA
            vg = xl.reshape(B, H // win, win, W // win, win, self.num_heads, self.head_dim).permute(0, 5, 1, 3, 2, 4,6).contiguous().\
                  reshape(B, self.num_heads, H * W // win**2, win**2, self.head_dim)
            # (B,H,W,C) -> (B,H//win,win,W//win,win,num_heads,head_dim) -> (B,num_heads,H//win,W//win,win,win,head_dim) -> (B,num_heads,H*W//win**2,win**2,head_dim)
            
            qkg = self.g_qk(self.norm_qk(
                    self.sr(xl.permute(0, 3, 1, 2).contiguous()).reshape(B, C, -1).permute(0, 2, 1).contiguous())). \
                    reshape(B, H * W // win**2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            # 难点 提取令牌的操作
            # sr 输出为 (B, C, H // win,W // win)
            # reshape (B,C, H * W // win ** 2) -> (B, H * W // win ** 2,C) 
            # norm_qk 输出为 (B,H * W // win ** 2,C) 
            # g_qk 输出为 (B, H * W // win ** 2,2 * C)
            # reshape (B, H * W // win**2, 2, num_heads, head_dim)
            # (2,B,num_heads,H * W // win**2,head_dim)
        
            atte_g = self.attend(qkg[0] @ qkg[1].transpose(-2, -1).contiguous() * self.scale)
            # atte_g.shape = (B, num_heads,H*W // win**2,H * W // win**2)
            xg = einsum('b h i j, b h j m c -> b h i m c', atte_g, vg)
            # 爱因斯坦求和约定 按照相应的规则进行求和
            # atte_g.shape = (B, num_heads,H*W // win**2,H * W // win**2)
            # vg.shape = (B,num_heads,H*W//win**2,win**2,head_dim)
            # 点乘操作 xg.shape = (B,num_heads,H*W//win**2,win ** 2,head_dim)
            xg = xg.reshape(B, self.num_heads, H // win, W // win, win, win, self.head_dim).permute(0, 2, 4, 3, 5, 1, 6).contiguous(). \
                    reshape(B, H, W, C)
            # (B,num_heads,H*W//win**2,win**2,head_dim) -> (B, num_heads,H // win, W // win, win, win, head_dim) ->(B,H // win,win,W//win,win,num_heads,head_dim) -> (B,H,W,C)
            xg = self.down(xg.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(B, N, C)
            x = xg.contiguous()
            
        if self.win_size == 1:
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            attn = self.attend((qkv[0] @ qkv[1].transpose(-2, -1).contiguous()) * self.scale)
            x = (attn @ qkv[2]).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)

        return x
    

class MergeAttention(BaseModule):

    def __init__(self, token, embed_dims, num_heads=8, attn_drop=0., proj_drop=0., rate=None, qkv_bias=True, dropout_layer=None, init_cfg=None):
        super(MergeAttention,self).__init__()
        if dropout_layer is None:
            dropout_layer = dict(type='DropPath', drop_prob=0.)

        self.rate = rate # 借助此变量选择相应窗口
        self.num_heads = num_heads
        self.head_dim = embed_dims // self.num_heads

        if rate == 8:
            self.num = 2 # 表示分支数目
            self.div = 2 # embed_dims // self.div 表示所映射的维度数目
            self.mapping = nn.Linear(embed_dims, embed_dims // self.div,bias = qkv_bias)
            
            num3 = self.num_heads // self.div
            self.norm3 = nn.LayerNorm(self.head_dim * num3)
            self.attn3 = DSA(
                token = token,
                embed_dims = self.head_dim * num3,
                num_heads = num3,
                win_size = 3,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                qkv_bias = qkv_bias,
                init_cfg = init_cfg)
            
            num5 = self.num_heads // self.div
            self.norm5 = nn.LayerNorm(self.head_dim * num5)
            self.attn5 = DSA(
                token = token,
                embed_dims = self.head_dim * num5,
                num_heads = num5,
                win_size = 5,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
            
        if rate == 4:
            self.num = 3 # 表示分支数目
            self.div = 4
            self.mapping = nn.Linear(embed_dims, embed_dims // self.div,bias = qkv_bias)

            num2 = self.num_heads // self.div
            self.norm2 = nn.LayerNorm(self.head_dim * num2)
            self.attn2 = DSA(
                token = token,
                embed_dims = self.head_dim * num2,
                num_heads = num2,
                win_size = 2,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                qkv_bias = qkv_bias,
                init_cfg = init_cfg)
            
            num3 = self.num_heads // self.div
            self.norm3 = nn.LayerNorm(self.head_dim * num3)
            self.attn3 = DSA(
                token=token,
                embed_dims = self.head_dim * num3,
                num_heads=num3,
                win_size = 3,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
            
            num5 = self.num_heads // self.div * 2
            self.norm5 = nn.LayerNorm(self.head_dim * num5)
            self.attn5 = DSA(
                token=token,
                embed_dims = self.head_dim * num5,
                num_heads=num5,
                win_size = 5,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
            
        if rate == 2:
            self.num = 3 # 表示分支数目
            self.div = 4
            self.mapping = nn.Linear(embed_dims, embed_dims // self.div,bias = qkv_bias)

            num2 = self.num_heads // self.div
            self.norm2 = nn.LayerNorm(self.head_dim * num2)
            self.attn2 = DSA(
                token = token,
                embed_dims = self.head_dim * num2,
                num_heads = num2,
                win_size = 2,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                qkv_bias = qkv_bias,
                init_cfg = init_cfg)
            
            num3 = self.num_heads // self.div
            self.norm3 = nn.LayerNorm(self.head_dim * num3)
            self.attn3 = DSA(
                token=token,
                embed_dims = self.head_dim * num3,
                num_heads=num3,
                win_size = 3,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
            
            num5 = self.num_heads // self.div * 2
            self.norm5 = nn.LayerNorm(self.head_dim * num5)
            self.attn5 = DSA(
                token=token,
                embed_dims = self.head_dim * num5,
                num_heads=num5,
                win_size = 5,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
            
        if rate == 1:
            self.num = 1
            self.div = 1
            self.mapping = nn.Linear(embed_dims, embed_dims // self.div,bias = qkv_bias)
            
            num1 = self.num_heads // self.div
            self.norm1 = nn.LayerNorm(self.head_dim * num1)
            self.attn1 = DSA(
                token=token,
                embed_dims = self.head_dim * num1,
                num_heads=num1,
                win_size = 1,
                attn_drop=attn_drop,
                proj_drop = proj_drop,
                qkv_bias=qkv_bias,
                init_cfg=init_cfg)
        

        self.proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims, bias=qkv_bias),
            nn.Dropout(proj_drop)
        )
        self.drop = build_dropout(dropout_layer)


    def forward(self,x,hw_shape):
        H,W = hw_shape
        B,N,C = x.shape
        x_map = self.mapping(x).contiguous()
        x_res = [] # 用来存放每一层的结果
        
        if self.rate == 8:
            x_run = x_map
            for i in range(self.num):
                if i == 0:
                    x_run = self.norm3(x_run)
                    x_run = self.attn3(x_run,hw_shape)
                if i == 1:
                    x_run = self.norm5(x_run)
                    x_run = self.attn5(x_run,hw_shape)
                
                x_res.append(x_run)
                if i != self.num - 1:
                    x_run =  x_run + x_map
                
        if self.rate == 4:
            x_run = x_map
            for i in range(self.num):
                if i == 0:
                    x_run = self.norm2(x_run)
                    x_run = self.attn2(x_run,hw_shape)
                if i == 1:
                    x_run = self.norm3(x_run)
                    x_run = self.attn3(x_run,hw_shape)
                if i == 2:
                    x_run = self.norm5(x_run)
                    x_run = self.attn5(x_run,hw_shape)
                
                x_res.append(x_run)
                if i < self.num - 2:
                    x_run =  x_run + x_map
                if i == self.num - 2:
                    x_run = torch.cat(x_res,dim = -1) + torch.cat([x_map,x_map],dim = -1)

        if self.rate == 2:
            x_run = x_map
            for i in range(self.num):
                if i == 0:
                    x_run = self.norm2(x_run)
                    x_run = self.attn2(x_run,hw_shape)
                if i == 1:
                    x_run = self.norm3(x_run)
                    x_run = self.attn3(x_run,hw_shape)
                if i == 2:
                    x_run = self.norm5(x_run)
                    x_run = self.attn5(x_run,hw_shape)
                
                x_res.append(x_run)
                if i < self.num - 2:
                    x_run =  x_run + x_map
                if i == self.num - 2:
                    x_run = torch.cat(x_res,dim = -1) + torch.cat([x_map,x_map],dim = -1)
        
        if self.rate == 1:
            x_run = self.norm1(x_map)
            x_run = self.attn1(x_run,hw_shape)
            x_res.append(x_run)
                
        x = torch.cat(x_res,dim = -1).contiguous()

        return self.drop(self.proj(x))



class NewBlock(BaseModule):

    def __init__(self,
                 token,
                 embed_dims,
                 num_heads,
                 rate,
                 feedforward_ratio,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'), # 表示标准化层的配置参数 config
                 with_cp=False,
                 init_cfg=None):

        super(NewBlock, self).__init__()

        self.init_cfg = init_cfg # 表示初始化的配置参数
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1] 
        # build_norm_layer函数返回一个元组，第二个元素返回标准化层实例
        self.attn = MergeAttention(
            token=token,
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            rate=rate,
            qkv_bias=qkv_bias,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN( # 前向反馈网络 用来对数据进行非线性激活，学习更加抽象的特征
            embed_dims=embed_dims,
            feedforward_channels=int(embed_dims * feedforward_ratio),
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):
        
        # 传播过程
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape) + identity # 残差链接
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        
        
        if self.with_cp and x.requires_grad: 
            x = cp.checkpoint(_inner_forward, x) 
            # 此时采用此种方法降低对显存的使用，不存储中间数据，本质上就是利用时间换空间
        else:
            x = _inner_forward(x)
        return x


# 用于确定相关位置 可以提升百分点
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        # conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.s = s

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous() # (B,C,H,W) -> (B,N,C)
        return x 

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class NewBlockSequence(BaseModule):

    def __init__(self,
                 token,
                 embed_dims,
                 num_heads,
                 rate,
                 feedforward_ratio,
                 depth, # depth表示该模块中有几个block
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., # 表示dropout层丢弃神经元的概率
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList() # 用来生成一个容器，存放所生成的各个block
        self.pos_block = PosCNN(embed_dims, embed_dims) # 暂不明确
        for i in range(depth):
            block = NewBlock(
                token=token,
                embed_dims=embed_dims,
                num_heads=num_heads,
                rate=rate,
                feedforward_ratio=feedforward_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for i, block in enumerate(self.blocks): # enumerate用来将可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            x = block(x, hw_shape)
            if i == 0:
                x = self.pos_block(x, hw_shape) # 暂不明确 初始化参数

        if self.downsample: # 下取样 patch_merging窗口合并操作 
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape # 最后一层不进行下取样操作


@MODELS.register_module()
class PWT_concat(BaseModule):

    def __init__(self,
                 pretrain_img_size=224, # 预设的图像大小为224*224
                 in_channels=3, # 输入通道数
                 embed_dims=64, # 嵌入层通道数
                 rates=(8, 4, 2, 1), # 表示不同进度下所采用的过程
                 patch_sizes=(7, 3, 3, 3), # 表示patch_embedding和patch_merging的卷积核的大小
                 mlp_ratio=(8, 8, 4, 4), # FFN模块中的线性层的通道数量选择
                 depths=(3, 4, 6, 5), # depths[i]表示每个模块下封装了多少个block
                 num_heads=(2, 4, 8, 16), # attention的分组头数
                 strides=(4, 2, 2, 2), # 表示patch_embeding中卷积层的步幅
                 paddings=(3, 1, 1, 1), # 表示patch_embeding中的卷积层的padding
                 out_indices=(0, 1, 2, 3), 
                 qkv_bias=True,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0., # 表示attention中dropout过程中丢弃神经元的概率
                 drop_path_rate=0.1, # 表示映射过程以及FFN中的dropout的丢弃神经元的概率
                 use_abs_pos_embed=False, # 表示是否使用pos_embeding用作位置编码
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(PWT_concat, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths) # 表示一共分多少stage
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        # assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed( # patch_embeding的设置
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        row = pretrain_img_size[0] // strides[0]
        # col = pretrain_img_size[1] // strides[0]
        # self.tokens = row * col
        if self.use_abs_pos_embed: # 设置位置编码矩阵
            patch_row = pretrain_img_size[0] // patch_sizes[0]
            patch_col = pretrain_img_size[1] // patch_sizes[0]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, embed_dims, patch_row, patch_col)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule # 设置随机深度衰减规则
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ] # item()返回对应的值，其精度更高

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging( # 设置每个模块的patch_merging模块，用于下取样操作
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=patch_sizes[i+1],
                    stride=strides[i + 1],
                    padding=paddings[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = NewBlockSequence(
                token= int(row // int(math.pow(2, i)) if row % int(math.pow(2, i)) == 0 else row // int(math.pow(2, i)) + 1), # 表示单一方向上的序列长度 不是整个的图像序列长度
                embed_dims=in_channels,
                num_heads=num_heads[i],
                rate=rates[i],
                feedforward_ratio=mlp_ratio[i],
                depth=depths[i],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], # 前缀和 depths=(3, 4, 6, 5)
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample: 
                in_channels = downsample.out_channels # 更新输入的通道数

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(PWT_concat, self).train(mode)
        self._freeze_stages()
    
    # 暂不明确 我认为该函数的主要作用是用来冻结某些参数
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False # 
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}') # 返回对象的属性值
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    # 初始化参数权重
    def init_weights(self):
        # logger = get_root_logger()
        if self.init_cfg is None:
            # logger.warn(f'No pre-trained weights for '
            #             f'{self.__class__.__name__}, '
            #             f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        
        # 初始化操作
        if self.use_abs_pos_embed:
            h, w = self.absolute_pos_embed.shape[1:3]
            if hw_shape[0] != h or hw_shape[1] != w:
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed,
                    size=hw_shape,
                    mode='bicubic',
                    align_corners=False).flatten(2).transpose(1, 2)
            else:
                absolute_pos_embed = self.absolute_pos_embed.flatten(
                    2).transpose(1, 2)
            x = x + absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return tuple(outs) # outs存放的是每一个stage的中间结果，可用于进一步分析 