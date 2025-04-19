import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
import copy
import requests
import json
import re
from copy import deepcopy
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from enum import auto, Enum
import numpy as np
from typing import Dict, Any, Union, List
import dataclasses
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...dataset import DATASET_TYPE, DATASET_MODALITY
from .utils import process_images, preprocess_multimodal_image, preprocess_multimodal_video, preprocess
from .utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN



class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(SiglipVisionTower, self).__init__()
        # super(SiglipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        
        # model_path = "google/siglip-so400m-patch14-384"
        self.unfreeze_mm_vision_tower = not getattr(args, 'freeze_vision_tower', True)
        # base_model_name, res, interp = model_path, 384, 576
        self.delay_load = delay_load
        res, interp = 384, 576
        self.vision_tower_name = vision_tower_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        self.is_loaded = False
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        # clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        # self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.config.hidden_size
        self._image_size = self.vision_tower.config.image_size
        self._patch_size = self.vision_tower.config.patch_size
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images=None, interpolate_token=576, forward_n_layers = -1, forward_nth_embeds = None,):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                # images.to(device=self.device, dtype=self.dtype),
                images.to(device=self.device, dtype=self.dtype) if images is not None else None,
                output_hidden_states=True,
                # forward_n_layers = forward_n_layers,
                # forward_nth_embeds = forward_nth_embeds,
            ).hidden_states[-1]
            interp_features = self.interpolate(image_features)
            return interp_features
        

    def forward(self, images=None, forward_n_layers = -1, forward_nth_embeds = None,):
        if type(images) is list:
            # image_features = [self._forward(image.unsqueeze(0)) for image in images]
            image_features = [self._forward(image.unsqueeze(0), forward_n_layers=forward_n_layers, forward_nth_embeds=forward_nth_embeds) for image in images]
        else:
            # image_features = self._forward(images)
            image_features = self._forward(images, forward_n_layers=forward_n_layers, forward_nth_embeds=forward_nth_embeds)

        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size

    @property
    def image_size(self):  # resolution
        # return self.config.image_size
        try:
            return self.config.image_size
        except:
            return self._image_size

    @property
    def patch_size(self):
        # return self.config.patch_size
        try:
            return self.config.patch_size
        except:
            return self._patch_size

    @property
    def num_patches_per_side(self):
        if self._interp_size is not None:
            return int(self._interp_size**0.5)
        try:
            return self.image_size // self.patch_size
        except:
            return self._num_patches_per_side

    @property
    def num_patches(self):
        if self._interp_size is not None:
            return self._interp_size
        try:
            return self.num_patches_per_side**2
        except:
            return self._num_patches
        


def get_3d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (n_images, image_height, image_width)
    return:
    pos_embed: [n_images, image_height, image_height, embed_dim]
    """
    if isinstance(image_size, int):
        image_size = [image_size] * 3
    else: # 3d
        grid_t_size, grid_h_size, grid_w_size = image_size[0], image_size[1], image_size[2]

    # grid_t = np.arange(grid_t_size, dtype=np.float32)
    # grid_h = np.arange(grid_h_size, dtype=np.float32)
    # grid_w = np.arange(grid_w_size, dtype=np.float32)
    # grid = np.meshgrid(grid_w, grid_t, grid_h) # (.shape=[t,w,h])
    # grid = np.stack(grid, axis=0)
    grid_t = torch.arange(grid_t_size, dtype=torch.float32)
    grid_h = torch.arange(grid_h_size, dtype=torch.float32)
    grid_w = torch.arange(grid_w_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_t, grid_w, grid_h) # (.shape=[t,w,h])
    grid = torch.stack(grid, dim=0)

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """ The embeddings parts for time-height-width will be [3/8, 3/8, 2/8].
    """
    assert embed_dim % 8 == 0
    # use 2/8 of dimensions to encode grid_t
    emb_t = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 2, grid[0])  # (T, H, W, D/8*2)
    # use 3/8 of dimensions to encode grid_h and grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 3, grid[1])  # (T, H, W, D/8*3)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 3, grid[2])  # (T, H, W, D/8*3)

    # emb = np.concatenate([emb_t, emb_h, emb_w], axis=-1)  # (T, H, W, D)
    emb = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # (T, H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (T, H, W)
    out: (T, H, W, D)
    """
    assert embed_dim % 2 == 0
    # omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    # out = np.einsum('thw,d->thwd', pos, omega)  # (T, H, W, D/2), outer product
    out = torch.einsum('thw,d->thwd', pos, omega)  # (T, H, W, D/2), outer product

    # emb_sin = np.sin(out)  # (T, H, W, D/2)
    # emb_cos = np.cos(out)  # (T, H, W, D/2)
    emb_sin = torch.sin(out)  # (T, H, W, D/2)
    emb_cos = torch.cos(out)  # (T, H, W, D/2)

    # emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (T, H, W, D)
    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (T, H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    # def __init__(
    #         self,
    #         num_queries,
    #         embed_dim,
    #         num_heads,
    #         kv_dim=None,
    #         norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #         adaptive=False,
    #         max_size=(70, 70,),
    # ):
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.num_queries = getattr(model_args, 'num_queries', kwargs.get('num_queries', 64))
        self.kv_dim = getattr(model_args, 'mm_resampler_visiondim', kwargs.get('mm_resampler_visiondim', None))
        self.embed_dim = getattr(model_args, 'mm_resampler_embeddim', kwargs.get('mm_resampler_embeddim'))
        self.num_heads = getattr(model_args, 'num_heads', kwargs.get('num_heads', None))
        # self.norm_layer = getattr(model_args, 'norm_layer', kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)))
        self.adaptive = getattr(model_args, 'adaptive', kwargs.get('adaptive', True))
        self.max_size = getattr(model_args, 'max_size', kwargs.get('max_size', (300, 24, 24,)))

        self.num_heads = self.embed_dim//128 if not self.num_heads else self.num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))

        if self.kv_dim is not None and self.kv_dim != self.embed_dim:
            self.kv_proj = nn.Linear(self.kv_dim, self.embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        # self.ln_q = self.norm_layer(self.embed_dim)
        # self.ln_kv = self.norm_layer(self.embed_dim)
        self.ln_q = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # self.ln_post = self.norm_layer(self.embed_dim)
        self.ln_post = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.proj = nn.Parameter((self.embed_dim ** -0.5) * torch.randn(self.embed_dim, self.embed_dim))

        self._set_3d_pos_cache(self.max_size)
        self.apply(self._init_weights)

    def _set_3d_pos_cache(self, max_size, device='cpu'):
        # pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, max_size).to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    # def _adjust_pos_cache(self, tgt_sizes, device):
    #     max_t = torch.max(tgt_sizes[:, 0]) if tgt_sizes[:, 0].nelement() !=0 else 0
    #     max_h = torch.max(tgt_sizes[:, 1]) if tgt_sizes[:, 1].nelement() !=0 else 0
    #     max_w = torch.max(tgt_sizes[:, 2]) if tgt_sizes[:, 2].nelement() !=0 else 0
    #     if max_t > self.max_size[0] or max_h > self.max_size[1] or max_w > self.max_size[2]:
    #         self.max_size = [max(max_t, self.max_size[0]), max(max_h, self.max_size[1]), max(max_w, self.max_size[2])]
    #         self._set_3d_pos_cache(self.max_size, device)
    
    def _adjust_pos_cache(self, max_thw_sizes, device):
        if max_thw_sizes[0] > self.max_size[0] or max_thw_sizes[1] > self.max_size[1] or max_thw_sizes[2] > self.max_size[2]:
            self.max_size = [max(max_thw_sizes[0], self.max_size[0]), max(max_thw_sizes[1], self.max_size[1]), max(max_thw_sizes[2], self.max_size[2])]
            self._set_3d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def forward(self, x, tgt_sizes=None):
    def forward(self, image_features: torch.tensor, tgt_size_range: list, *args, **kwargs):
        """ Forward resampler using 3D positional encoding. Pass right-open ranges [[t_star,t_end),[w_start,w_end),[h_start,h_end)] for 'tgt_size_range' if input videos, elsewise pass [[w_start,w_end),[h_start,h_end)] for input images.
        """
        # tgt_sizes = torch.tensor([tgt_size]*image_features.shape[0], dtype=torch.int32) # Assume all feat-maps in same shape
        tgt_size_range = [[0,_] if isinstance(_,int) else _ for _ in tgt_size_range] # convert to range
        if len(tgt_size_range) == 2:
            tgt_size_range = [[0,1], tgt_size_range[0], tgt_size_range[1]]
        elif len(tgt_size_range) == 3: # iunput videos
            image_features = image_features.view(image_features.shape[0], -1, image_features.shape[-1])
        B, L, D = image_features.shape

        # tgt_sizes = torch.tensor(tgt_size, device=image_features.device, dtype=torch.int32).unsqueeze(0).repeat(B,1)
        tgt_sizes_range = torch.tensor(tgt_size_range, device=image_features.device, dtype=torch.int32).unsqueeze(0).repeat(B,1,1)
        tgt_sizes = torch.tensor([_[1]-_[0] for _ in tgt_size_range], device=image_features.device, dtype=torch.int32).unsqueeze(0).repeat(B,1)

        # tgt_sizes = torch.ones(image_features.shape[0], 3, device=image_features.device, dtype=torch.int32)
        # for b in range(image_features.shape[0]):
        #     tgt_sizes[b] = torch.tensor(tgt_size, dtype=torch.int32)
        # B, L, D = image_features.shape

        x = image_features
        assert x.shape[0] == tgt_sizes_range.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1] * tgt_sizes[:, 2]

        # self._adjust_pos_cache(tgt_sizes, device=device)
        self._adjust_pos_cache([_[1] for _ in tgt_size_range], device=device) # -1 for right-open

        # max_patch_len = torch.max(patch_len)
        max_patch_len = torch.max(patch_len) if patch_len.nelement() !=0 else 0
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_t, tgt_h, tgt_w = tgt_sizes[i]
            range_t, range_h, range_w = tgt_sizes_range[i]
            # pos_embed.append(self.pos_embed[:tgt_t, :tgt_h, :tgt_w, :].reshape((tgt_t * tgt_h * tgt_w, -1)).to(dtype))  # n_images * patches * D
            pos_embed.append(self.pos_embed[range_t[0]:range_t[1], range_h[0]:range_h[1], range_w[0]:range_w[1], :].reshape((tgt_t * tgt_h * tgt_w, -1)).to(dtype))  # n_images * patches * D
            key_padding_mask[i, patch_len[i]:] = True

        # pos_embed = torch.nn.utils.rnn.pad_sequence(
        #     pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        if pos_embed!=[]:
            pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
        else:
            pos_embed, key_padding_mask = torch.zeros_like(x, device=x.device), None
        pos_embed = pos_embed.to(x.device)
        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

    @property
    def config(self):
        return {
            "mm_resampler_type": "qformer",
            "num_queries": self.num_queries,
            "kv_dim" : self.kv_dim,
            "embed_dim" : self.embed_dim,
            "num_heads" : self.num_heads,
            # "norm_layer" : self.norm_layer,
            "adaptive" : self.adaptive,
            "max_size" : self.max_size,
        }

    @property
    def hidden_size(self):
        return self.embed_dim

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}




def find_segments(A):
    segments = []
    pre, cur = 0, 0
    while cur < len(A):
        if cur==len(A)-1 or (A[cur+1]!=0):
            segments.append((pre, cur+1)) # add tuple [closed-left open-right)
            pre = cur+1
        cur += 1
    return segments

def sample_gumbel(shape, eps=1e-20, dtype=torch.bfloat16):
    U = torch.rand(shape, dtype=dtype)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, lr_gumbel):
    # y = logits + sample_gumbel(logits.size(), dtype=logits.dtype)
    y = logits + sample_gumbel(logits.size(), dtype=logits.dtype) * lr_gumbel
    return F.softmax(y / temperature, dim=-1)


# def gumbel_softmax(logits, temperature, topk=1, hard=False):
def gumbel_softmax(logits, temperature, topk=1, lr_gumbel=0.1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    shape = logits.shape
    y = gumbel_softmax_sample(logits, temperature, lr_gumbel) # (1, N, 2)
    y = y[:, :, 1]

    _, ind = y.topk(k=topk, dim=-1) # Qiji: changed to N-hot
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind, 1)
    y_hard = (y_hard - y).detach() + y
    return y, y_hard


class Cubing(nn.Module):
    def __init__(self,
                cubing_type,
                vision_dim,
                vision_toks_len,
                embed_dim=256,
                window_size=3,
                mm_use_thumbnail=True,
                forward_n_layers=-1,
                lm_dim = 4096,
                **kwargs) -> None:
        super().__init__(**kwargs)

        self.cubing_type = cubing_type
        self.vision_dim = vision_dim
        self.embed_dim = vision_dim if embed_dim is None else embed_dim
        self.mm_use_thumbnail = mm_use_thumbnail
        self.forward_n_layers = forward_n_layers


        self.agg_frame_fn = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim),
        )

        self.proj_fn = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.vision_dim),
            nn.GELU(),
            nn.Linear(self.vision_dim, 2),
            )

        self.thumbnail_fn = nn.Sequential(
            nn.AvgPool2d(kernel_size=(9,1), stride=(9,1)),
            nn.Linear(self.vision_dim, lm_dim)
        )

        

    def forward(self, vision_tower, resampler, images, tgt_size, videos_bound, temperature=0.5, FPQ=5, lr_gumbel=0.1):
        """
          Params:
            @FPQ: the average number of frames per cube.
            @lr_gumbel: the weight of the gumbel_noise, for annealing.
        """
        # bs, L, D = image_features.shape
        bs = len(images)
        # h, w = tgt_size

        # debug_feats = None
        cube_bound = [] # [(bounds of cubes for video 1), ...]
        pre_e = 0
        list_feats = []
        list_z = []
        for i, (vs, ve) in enumerate(videos_bound):
            if vs > pre_e:
                # pre_img_feats = image_features[pre_e: vs]
                pre_img_feats = vision_tower(images[pre_e: vs])
                pre_img_feats = resampler(pre_img_feats, tgt_size)
                list_feats.append(pre_img_feats)

            # video = image_features[vs: ve]
            video = vision_tower(images[vs: ve], self.forward_n_layers) # Use first n-layers
            # Resnet to classify each frame gradually
            bf = len(video)

            # Momentum
            # video = self.agg_frame_fn(vid_feats) # Before or After for project vit feats
            vid_feats_momentum = [video[1] - video[0]]
            alpha = 0.8
            for ii in range(2,bf):
                vid_feats_momentum.append(alpha*(video[ii]-video[ii-1]) + (1-alpha)*(vid_feats_momentum[-1]))
            vid_feats = torch.stack(vid_feats_momentum, dim=0) # (bf-1, 576, 1024)

            vid_feats = self.agg_frame_fn(vid_feats) # Before or After for project vit feats
            vid_feats = vid_feats.mean(dim=1)

            # vid_feats = torch.mean(video, dim=1, keepdim=False)
            z = self.proj_fn(vid_feats) # (bf-1, 2)
            list_z.append(z)
            # print(f"********\n Z after projection: {z.tolist()}\n sp_rank: {self.sp_rank}")

            num_cubes = max(round(bf/FPQ)-1, 1) # -1 to exclude beginning
            # print(f"********\n Number of cubes: {num_cubes} \n sp_rank: {self.sp_rank}")
            print(f"#### [lr_gumbel = {lr_gumbel}]")
            z, z_hard = gumbel_softmax(z.unsqueeze(0), temperature, topk=num_cubes, lr_gumbel=lr_gumbel)
            # print(f"********\n Z after Gumbel_softmax: {z.tolist()}\n sp_rank: {self.sp_rank}")
            z, z_hard = z.squeeze(0), z_hard.squeeze(0)

            # Force add 1 to beginning
            pad_z, pad_z_hard = torch.ones(1,dtype=z.dtype,device=z.device), torch.ones(1,dtype=z_hard.dtype,device=z_hard.device)
            z, z_hard = torch.cat([pad_z,z]), torch.cat([pad_z_hard, z_hard]) # (bf, )

            # Decide the cubes boundaries based on Gumbel-Sampling
            bounds = find_segments(z_hard)
            # print(f"********\n Cube bounds: {bounds}")
            cube_bound.append(bounds)

            vid_feats = []
            for ii,(s,e) in enumerate(bounds):
                feat = video[s:e]
                feat = resampler(feat.unsqueeze(0), tgt_size_range=[[s,e], [0,tgt_size[0]], [0,tgt_size[1]]]) # feat: (1, bf*576, d) ## Change this for using 3D resampler
                vid_feats.append(feat)
            vid_feats = torch.cat(vid_feats, 0)

            if self.mm_use_thumbnail:
                if self.forward_n_layers >= 0:
                    indexs = torch.nonzero(z_hard).view(-1)
                    video_thumb = video[indexs]
                    video_thumb = vision_tower(forward_n_layers=self.forward_n_layers, forward_nth_embeds = video_thumb)
                    video = video.scatter(0, indexs.view(-1,1,1).repeat(1,*video.shape[1:]), video_thumb)

                bound_feats = video * z_hard.unsqueeze(1).unsqueeze(1) / z_hard.sum().detach().item() # video.shape = (128,576,4096)
                bound_feats = torch.sum(bound_feats, dim=0, keepdim=True)
                # SUPPORT for sequence_parallel ! Only add thumbnail feature at the first chunk of sequence_parallel
                bound_feats = self.thumbnail_fn(bound_feats)
                vid_feats = torch.cat([bound_feats, vid_feats], 0) # (bf+1, 64, 4096)
                # vid_feats = vid_feats[int(self.sp_rank>0):] # Add thumbnail only at the first chunk
                # debug_feats = vid_feats

            list_feats.append(vid_feats)
            pre_e = ve

        if pre_e != bs:
            post_img_feats = vision_tower(images[pre_e:])
            # post_img_feats = resampler(image_features[pre_e:], tgt_size)
            post_img_feats = resampler(post_img_feats, tgt_size)
            list_feats.append(post_img_feats)
        
        image_features = torch.vstack(list_feats)

        if self.mm_use_thumbnail:
            # Update with thumbnail only on the first chunk
            videos_bound = [(vb[0]+i, vb[1]+i+1) for i,vb in enumerate(videos_bound)]
            cube_bound = [[(0,1)]+[(cb[0]+1, cb[1]+1) for cb in vcb] for vcb in cube_bound] # (0,0) for thumbnail seg
        return image_features, videos_bound, cube_bound, list_z

    @property
    def config(self):
        return {
            "cubing_type": self.cubing_type,
            "mm_use_thumbnail": self.mm_use_thumbnail
        }

    @property
    def hidden_size(self):
        return self.embed_dim




class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = SiglipVisionTower(config.mm_vision_tower, config, delay_load=False)
            # self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.vision_resampler = Resampler(config, mm_resampler_embeddim=self.config.hidden_size, mm_resampler_visiondim=self.vision_tower.hidden_size)
            self.mm_projector = IdentityMap()
            # self.vision_resampler.mm_projector = self.mm_projector
            
            self.cubing_type = getattr(config, 'mm_cubing', 'identity')
            self.cubing = Cubing(
                config.mm_cubing,
                self.vision_tower.hidden_size,
                self.vision_tower.num_patches,
                lm_dim = self.config.hidden_size,
                mm_use_thumbnail=config.mm_use_thumbnail,
                forward_n_layers=config.cubing_vit_forward_n_layers,
            )

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    



def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor




def update_placeholders_by_cube(cube_bound, video_bound, input_ids, attention_mask, labels, position_ids=None, seqlens_in_batch=None, IMID=-200, PADID=128256, NUM_TOKENS_PER_IMAGE=64):
    """ 1. input_ids/labels: merge multiple <image> in one cube, 2. position_ids: merge and shift, 3. attention_mask: merge and shift.
      Assume video bounds, e.g., [[0, 4), [4, 8), ...]
      Assume the cube bounds are consecutive not overlaps, e.g., [[[0, 3), [3, 4)],  [[0, 4)], ...]
    """
    if not video_bound or not cube_bound:
        # print(f"video_bound: {video_bound} cube_bound: {cube_bound}, input_ids-shape: {input_ids.shape}, position_ids-shape: {position_ids.shape}, position_ids: {position_ids}")
        return input_ids, labels, attention_mask, position_ids, seqlens_in_batch
    # print(f"********** BEFORE UPDATE\n cube_bound: {cube_bound},\n video_bound: {video_bound},\n num_imgs_in_inputs: {(input_ids==-200).sum()},\n effective_len: {(position_ids!=-1).sum()},\n seqlens_in_batch: {seqlens_in_batch}\n input_ids: {input_ids.tolist()}\n labels: {labels.tolist()}")

    bs, seq_len = input_ids.shape
    new_input_ids, new_labels, new_attention_mask, new_position_ids = [], [], [], []
    b, g = 0, 0
    try:
        while b < input_ids.size(0):
            cur_input_ids, cur_labels, cur_attention_mask, cur_position_ids = [], [], [], []
            i, p, pos_off = 0, 0, 0
            # while i < input_ids.size(1):
            while i < input_ids.size(1) and input_ids[b,i]!=PADID: # WARNING: Set terminating line as PADID!
                if input_ids[b, i] == IMID:
                    cbounds = None
                    for ii, vbound in enumerate(video_bound):
                        if vbound[0] <= g < vbound[1]: # is a video frame
                            break
                    if vbound[0] <= g < vbound[1]:
                        cbounds = cube_bound[ii] # found N bounds of cubes
                    # Merge
                    if cbounds is not None:
                        assert g == cbounds[0][0]+ vbound[0], f"The image number in input_ids and in cube_bound do not match!\n input_ids:{input_ids.tolist()}\n video_bound: {video_bound}\n cube_bound: {cube_bound}\n g: {g}"
                        for cb in cbounds:
                            assert cb[0]+vbound[0] <= g < cb[1] + vbound[0]
                            im_added = False
                            # Merge
                            # while g < cb[1] + vbound[0] and i < input_ids.size(1):
                            while g < cb[1] + vbound[0] and i < input_ids.size(1) and input_ids[b,i]!=PADID: # WARNING: Set terminating line as PADID!
                                # Merge only IMG tokens and ALL intermediate tokens between cb[0] and cb[1]
                                if (input_ids[b, i] == IMID and not im_added) or (input_ids[b, i] != IMID and g<= cb[0] + vbound[0]): # tokens before left boundary
                                    cur_input_ids.append(input_ids[b,i])
                                    cur_labels.append(labels[b,i])
                                    cur_attention_mask.append(attention_mask[b,i])
                                    if input_ids[b,i] == IMID:
                                        im_added = True
                                        g += 1 # accumulate frame
                                        if position_ids is not None:
                                            pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                                            cur_position_ids.extend([ _- pos_off for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]) # update positions of image tokens
                                            if any([ _- pos_off<-1 for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]):
                                                assert 1==2
                                            p += NUM_TOKENS_PER_IMAGE
                                    else:
                                        if position_ids is not None:
                                            pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                                            cur_position_ids.append(position_ids[b, p] - pos_off)
                                            if position_ids[b, p] - pos_off<-1:
                                                assert 1==2
                                            p += 1
                                elif i != input_ids.size(1)-1: # Reduce here
                                # Reduce when i is not at the last position, elsewise will end the 2nd while-loop based on i
                                # else: # Reduce here
                                    if input_ids[b, i] == IMID:
                                        g += 1 # accumulate frame
                                        if position_ids is not None:
                                            pos_off = pos_off+sum([position_ids[b,p+_]-position_ids[b,p+_-1] for _ in range(1,NUM_TOKENS_PER_IMAGE)])+1 # update with deltas
                                            p += NUM_TOKENS_PER_IMAGE
                                    else:
                                        if position_ids is not None:
                                            pos_off = pos_off+(position_ids[b,p+1]-position_ids[b,p])
                                            p += 1
                                i +=1
                    else:
                        cur_input_ids.append(input_ids[b,i])
                        cur_labels.append(labels[b,i])
                        cur_attention_mask.append(attention_mask[b,i])
                        i += 1
                        g += 1 # accumulate image
                        if position_ids is not None:
                            pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                            cur_position_ids.extend([ _- pos_off for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]) # update positions of image tokens
                            if any([ _- pos_off<-1 for _ in position_ids[b, p: p+NUM_TOKENS_PER_IMAGE]]):
                                assert 1==2
                            p += NUM_TOKENS_PER_IMAGE
                else:
                    cur_input_ids.append(input_ids[b,i])
                    cur_labels.append(labels[b,i])
                    cur_attention_mask.append(attention_mask[b,i])
                    i += 1
                    if position_ids is not None:
                        pos_off = 0 if position_ids[b, p] <= 0 else pos_off
                        cur_position_ids.append(position_ids[b, p] - pos_off)
                        if position_ids[b, p] - pos_off <-1:
                            assert 1==2
                        p += 1
            b+=1
            new_input_ids.append(torch.stack(cur_input_ids))
            new_labels.append(torch.stack(cur_labels))
            new_attention_mask.append(torch.stack(cur_attention_mask))
            if position_ids is not None:
                _i = len(cur_position_ids)-1
                while _i>=0 and cur_position_ids[_i]==-1: _i-=1
                cur_position_ids = cur_position_ids[:_i+1] # remove excess paddings at the end
                new_position_ids.append(torch.stack(cur_position_ids))
    except BaseException as e:
        print(f"{e}")
        print(f"---------\n b: {b},\n pos_off:{pos_off},\n p: {p},\n i: {i},\n video_bound: {video_bound},\n cube_bound: {cube_bound},\n seqlens_in_batch: {seqlens_in_batch},\n input_ids: {input_ids.tolist()},\n position_ids: {position_ids.tolist() if position_ids is not None else None},\n", file=open(f"tmp_log_{input_ids.device}.log", 'w'))
        assert 1==2
    new_input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=PADID)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=False)
    new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1) if position_ids is not None else position_ids
    new_seqlens_in_batch = seqlens_in_batch
    if seqlens_in_batch is not None:
        assert new_position_ids is not None
        
        sp_new_position_ids = new_position_ids
        # update
        new_seqlens_in_batch = []
        for b, posids in enumerate(sp_new_position_ids):
            prev_pid = posids[0]
            for pid in posids:
                if pid == -1: continue
                if pid > prev_pid:
                    new_seqlens_in_batch[-1] +=1
                else: # new sample in batch
                    new_seqlens_in_batch.append(1)
        new_seqlens_in_batch = torch.tensor(new_seqlens_in_batch, dtype=seqlens_in_batch.dtype, device=seqlens_in_batch.device)
    return new_input_ids, new_labels, new_attention_mask, new_position_ids, new_seqlens_in_batch



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_resampler(self):
        return self.get_model().vision_resampler



    def encode_images(self, images, videos_bound, tgt_size, is_cubing=True, temperature=0.5, FPQ=5, lr_gumbel=0.1, prompts=None, long_video=False):
        cube_bound, list_z = None, []
        if is_cubing:
            image_features, videos_bound, cube_bound, list_z = self.get_model().cubing(self.get_model().get_vision_tower(), self.get_model().vision_resampler, images, tgt_size, videos_bound, temperature, FPQ, lr_gumbel)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            B, L, D = image_features.shape # bs*imgcount, seq_len, d_model
            image_features = self.get_model().vision_resampler(image_features, tgt_size=tgt_size)
            image_features = self.get_model().mm_projector(image_features)

        # return image_features, cube_losses
        return image_features, videos_bound, cube_bound, list_z

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities, image_sizes=None,prompts=None, imidx_in_multi=None,
        seqlens_in_batch=None, videos_bound=None, lr_gumbel=0.1,
    ):
        sp_degree = -1
        sp_rank = -1


        vision_tower = self.get_vision_tower()
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if vision_tower is None or images is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, seqlens_in_batch, [None]

        # pre-process images for long video
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        if isinstance(modalities, str):
            modalities = [modalities]

        image_idx_in_batch, video_idx_in_batch = [], []
        for _ in range(len(modalities)):
            # if modalities[_] != "video":
            if modalities[_] == "image":
                image_idx_in_batch.append(_)
            elif modalities[_] == 'video':
                video_idx_in_batch.append(_)
        if type(images) is list or images.ndim == 5:
            # not reseshape for long video

            # if not long_video:
            images_list, _videos_bound, img_counts_batch, vid_counts_batch = [], [], [0]*len(images), [0]*len(images)
            prev = 0
            for i, image in enumerate(images):
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
                if modalities[i] == 'video':
                    _videos_bound.append([prev, prev+image.shape[0]])
                    vid_counts_batch[i] += 1
                else:
                    img_counts_batch[i] += image.size(0)
                prev += image.shape[0]

            if videos_bound is None:
                videos_bound = _videos_bound
            try:
                concat_images = torch.cat(images_list, dim=0)
            except Exception as e:
                print(e)
                for _ in images_list:
                    print(_.shape)
                import pdb
                pdb.set_trace()

            is_cubing =  self.get_model().config.is_cubing
            tgt_size = (self.get_vision_tower().num_patches_per_side, )*2
            image_features, videos_bound, cube_bound, list_z = self.encode_images(concat_images, videos_bound, tgt_size, is_cubing, temperature=0.5, lr_gumbel=lr_gumbel)
            # Update image placeholders in input_ids, labels based on cubes
            input_ids_debug, attention_mask_debug, labels_debug, position_ids_debug, seqlens_in_batch_debug = input_ids, attention_mask, labels, position_ids, seqlens_in_batch
            if is_cubing:

                input_ids, labels, attention_mask, position_ids, seqlens_in_batch  = \
                    update_placeholders_by_cube(cube_bound, videos_bound, input_ids, attention_mask, labels, position_ids, seqlens_in_batch,PADID=self.config.pad_token_id)
                split_sizes = 1 # Qiji: FIXME
            else:   
                split_sizes = [image.shape[0] for image in images_list]

            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # new_image_features.append(image_feature.flatten(0, 1))
                    new_image_features.extend(torch.split(image_feature, 1, dim=0))
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):

                    # For video
                    if image_idx not in image_idx_in_batch:
                        # new_image_features.append(image_feature.flatten(0, 1))
                        new_image_features.extend(torch.split(image_feature, 1, dim=0))
                        continue

                    if image_feature.shape[0] > 1:
                        image_feature_multi = []
                        n_multi = len(imidx_in_multi[image_idx])
                        for ii, imid in enumerate(imidx_in_multi[image_idx]):
                            base_image_feature = image_feature[imid]
                            image_feature = image_feature[imid+1: imidx_in_multi[image_idx][ii+1] if ii<n_multi-1 else 1000]
                            height = width = int(self.get_vision_resampler().num_queries ** (1/2))
                            if image_aspect_ratio == "anyres":
                                from .utils import get_anyres_image_grid_shape
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx][imid], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                image_feature = image_feature.view(2, 2, height, width, -1)

                            if "maxpool2x2" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = nn.functional.max_pool2d(image_feature, 2)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            elif "unpad" in mm_patch_merge_type:
                                # import pdb; pdb.set_trace()
                                #
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                # image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = unpad_image(image_feature, image_sizes[image_idx][imid])
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            if "nobase" in mm_patch_merge_type:
                                pass
                            else:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            image_feature_multi.append(image_feature)
                        image_feature = torch.cat(image_feature_multi, dim=0)
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                # image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            image_features = new_image_features
        else:
            # image_features = self.encode_images(images)
            image_features, cube_bound = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        # for _, inids in enumerate(input_ids):
        #     print(f"[Inputs_{_} of {len(input_ids)}]:{self.tokenizer.decode([x if x != -200 else 567 for x in inids])}")

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_labels = labels[batch_idx]

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                if cur_image_features.ndim == 3:
                    cur_image_features = cur_image_features.squeeze(0)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                try:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                except:
                    import pdb
                    pdb.set_trace()
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                # if sp_degree > 1 and i == 0 and sp_rank != 0:  # Handle sequence parallelism
                #     cur_input_ids_noim.append(cur_input_ids[0:0]) # chunks in rest start with <image>
                #     cur_labels_noim.append(cur_labels[0:0])
                #     continue
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except:
                        print(f"cur_input_ids: {cur_input_ids.tolist()}\n image_features-shape: {len(image_features)}\n modalities: {modalities}\n cur_image_idx: {cur_image_idx}\n videos_bound: {videos_bound}\n cubes_bound: {cube_bound}\n concat_images-shape: {concat_images.shape}\n input_ids: {input_ids}\n input_ids_debug: {input_ids_debug.tolist()}\n labels_debug:{labels_debug.tolist()}\n position_ids_debug:{position_ids_debug.tolist()}\n seqlens_in_batch_debug:{seqlens_in_batch_debug.tolist()}\n attention_mask_debug: {attention_mask_debug.tolist()}")
                        raise BaseException
                    if cur_image_features.ndim == 3:
                        hidden_size = cur_image_features.shape[-1]
                        cur_image_features = cur_image_features.reshape(-1, hidden_size)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                import warnings
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # import pdb; pdb.set_trace()

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        gpu_rank = new_input_embeds.device.index if new_input_embeds.is_cuda else None
    

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            seqlens_in_batch,
            list_z
        )


    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):

        # kentang-mit@: reorder and repack (reduce computation overhead)
        # requires transformers replacement.
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0
        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # each item: num_tokens x num_channels
                # remove padding on-the-fly
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                # each item: num_tokens
                # remove padding on-the-fly
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))
                # The current batch is too long. We will start a new batch.
                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
            # Mask the first token in the labels for every sample
            # cur_labels[-1][0] = IGNORE_INDEX

        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=self.model.config.pad_token_id
        )

        new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1)

        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        ## yunhao: it's currently a workaround to avoid errors for seq_len < 100
        new_attention_mask = new_position_ids.ne(-1)
        # sanity check
        assert new_attention_mask.sum() == attention_mask.sum()

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )





class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen"
    

class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

def _get_unpad_data(attention_mask: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if hasattr(_get_unpad_data, "seqlens_in_batch"):
        seqlens_in_batch = _get_unpad_data.seqlens_in_batch
    else:
        seqlens_in_batch = torch.sum(attention_mask, dim=1)

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch

def set_seqlens_in_batch(seqlens_in_batch: torch.Tensor) -> None:
    _get_unpad_data.seqlens_in_batch = seqlens_in_batch


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        # import pdb; pdb.set_trace()
        Qwen2ForCausalLM.__init__(self, config)
        self.model = LlavaQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Qiji: add for SP
        transformers.modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        imidx_in_multi: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        videos_bound: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            lr_gumbel = kwargs.get('lr_gumbel', 0.1)
            # (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch, debug_feats) = self.prepare_inputs_labels_for_multimodal(
            # (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch) = self.prepare_inputs_labels_for_multimodal(
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch, list_z) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, prompts, imidx_in_multi,
                seqlens_in_batch, videos_bound, lr_gumbel
            )

        # support_packing = "seqlens_in_batch" in inspect.signature(self.model.forward).parameters

        if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
        set_seqlens_in_batch(seqlens_in_batch)

        # if self.training and support_packing:
        need_repack = kwargs.get('need_repack', False)
        # if self.training and support_packing and need_repack:
        # if self.training and support_packing and need_repack and inputs_embeds is not None:
        if self.training and need_repack and inputs_embeds is not None:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            if sorted_seqlens_in_batch is not None:
                set_seqlens_in_batch(sorted_seqlens_in_batch)
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            # sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            sorted_seqlens_in_batch = attention_mask.sum(-1).int() if need_repack else seqlens_in_batch
            new_input_ids = input_ids

        # print(f"RRRRRR\n new_attention_mask: {new_attention_mask}\n new_position_ids:{new_position_ids}\n new_labels:{new_labels}")
        outputs = super().forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

        

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        imidx_in_multi: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        videos_bound: Optional[List[List[int]]] = None,
        tokenizer=None,
        terminators=['<|eot_id|>'],
        llm_device: Optional[torch.device] = None,
        lr_gumbel: Optional[int] = 0.0, # Qiji: default no Gumbel noise
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            labels = input_ids.clone()
            # (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, cube_losses) = self.prepare_inputs_labels_for_multimodal(
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, seqlens_in_batch, list_z) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, prompts, imidx_in_multi,
                seqlens_in_batch, videos_bound, lr_gumbel
            )

        # For inference model on different devices
        input_ids = None
        if llm_device is not None:
            inputs_embeds = inputs_embeds.to(llm_device)
            attention_mask = attention_mask.to(llm_device) if attention_mask is not None else None
            input_ids = torch.ones((inputs_embeds.shape[0], 0), dtype=torch.long, device=llm_device)

        output_ids = super().generate(
            input_ids = input_ids,
            # input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            seqlens_in_batch= seqlens_in_batch,
            **kwargs,
        )


        # decode text
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in terminators]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_token_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    
    @torch.no_grad()
    def chat(self,
            image,
            msgs, # [[{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]]
            modalities, # ['image', 'video', ..]
            tokenizer,
            image_processor,
            max_new_tokens=2048,
            # seq_parallel_size=1,
            dtype=None,
            **kwargs,
    ) -> Tuple[str, list, list]:
        """ Given user messages composed of interleaved image-prompt,
              respond the model output, past_key_values, and vision_hidden_states.
            Note that we assume each conversation refer to only one modality, either n images or a video!
          Params:
            @msgs: [[{'from': 'human', 'value': (Image, Image, 'Hello.')}, {'from': 'gpt', 'value': 'Hi!'}], ..]
            @image: [[Image, ], ..] or None
        """
        generation_config = {
            "tokenizer": tokenizer,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "terminators": [tokenizer.eos_token]
        }
        
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image
        
        if batched is False:
            images_list, msgs_list, modalities = [images_list], [msgs_list], [modalities]
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        # batch = []
        bsz_inputs, bsz_images, bsz_image_sizes, bsz_imidx_in_multi = [], [], [], []
        for i, (image, msgs, modality) in enumerate(zip(images_list, msgs_list, modalities)):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            if image is not None and isinstance(copy_msgs[0]["value"], str):
                copy_msgs[0]["value"] = [image, copy_msgs[0]["value"]] # Add image to form a tuple value

            cut = 2
            copy_msgs_rou = [copy_msgs[_:_+cut] for _ in range(0, len(copy_msgs), cut)]
            timestamps = None
            msgs_images = []
            for ii, rou_msgs in enumerate(copy_msgs_rou):
                rou_images = []
                for iii, msg in enumerate(rou_msgs):
                    role = msg["from"]
                    value = msg["value"]
                    timestamps = msg.get('timestamps', None) if 'timestamps' in msg else timestamps
                    assert role in ["human", "gpt"]
                    if iii == 0:
                        assert role == "human", "The role of first msg should be human"
                    if isinstance(value, str):
                        value = [value]
                    cur_msgs = []
                    for c in value:
                        if isinstance(c, Image.Image) and modality=='image':
                            cur_msgs.append(DEFAULT_IMAGE_TOKEN)
                            rou_images.append(c)
                        elif isinstance(c, np.ndarray) and c.ndim==4 and modality=='video':
                            cur_msgs.append(DEFAULT_IMAGE_TOKEN)
                            rou_images.extend([Image.fromarray(f) for f in c])
                        elif isinstance(c, str):
                            c = c.replace(DEFAULT_IMAGE_TOKEN, '')
                            cur_msgs.append(c)
                    msg["value"] = "\n".join(cur_msgs)
                msgs_images.extend(rou_images)

            # Process a conversations with multi-rounds
            images, conversations, image_sizes, imidx_in_multi = None, None, None, []
            image_sizes = [img.size for img in msgs_images]
            setattr(self.model.config, 'is_multimodal', True)
            if modality == 'image':
                images, pathnums_imgs = process_images(msgs_images, image_processor, self.model.config, return_pathnums=True)
                imidx_in_multi = np.cumsum([0] + [images[_].shape[0] for _ in range(len(images)-1)]).tolist()
                conversations = preprocess_multimodal_image([copy_msgs], self.model.config, pathnums_imgs)
                if isinstance(images, list):
                    images = torch.cat(images, dim=0)
            elif modality == 'video':
                images = process_images(msgs_images, image_processor, self.model.config, image_aspect_ratio='original', return_pathnums=False)
                imidx_in_multi = list(range(len(imidx_in_multi), len(imidx_in_multi)+len(images))) # Be careful
                conversations = preprocess_multimodal_video([copy_msgs], self.model.config, frame_timestamps=timestamps, nframes=len(images),is_cubing=True, add_thumbnail=self.model.config.mm_use_thumbnail)
            images = images.view(-1, *images.shape[-3:])
            if dtype is not None:
                images = images.to(dtype=dtype)
            
            input_ids = preprocess(
                conversations,
                tokenizer,
                has_image=len(images) > 0,
                prompt=None,
                build_labels=False)['input_ids'][0]
            # labels = input_ids.clone()
            
            # Collect for a batch
            bsz_inputs.append(input_ids)
            bsz_images.append(images)
            bsz_image_sizes.append(image_sizes)
            bsz_imidx_in_multi.append(imidx_in_multi)
            # batch.append({'input_ids':input_ids, 'labels':labels, 'images':images,'modality':modality, 'image_sizes':image_sizes, 'imidx_in_multi':imidx_in_multi})

        # Collator for a batch
        batch = {}
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            bsz_inputs,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)[:, :tokenizer.model_max_length].to(torch.device('cuda:0'))
        if len(bsz_images)>0:
            if all(x is not None and x.shape == bsz_images[0].shape for x in bsz_images) and len(bsz_images) > 1:
                batch['images'] = torch.stack(bsz_images).to(torch.device('cuda:0'))
            else:
                batch['images'] = [img.to(torch.device('cuda:0')) for img in bsz_images]
        batch['modalities'] = modalities
        batch['attention_mask'] = batch['input_ids'].ne(tokenizer.pad_token_id)
        batch['image_sizes'] = bsz_image_sizes
        batch['imidx_in_multi'] = bsz_imidx_in_multi
        # batch = batch_to(batch, compute_type=dtype, device=self.device)

        return self.generate(**batch, **generation_config, **kwargs)






from transformers import BitsAndBytesConfig, AutoTokenizer, AutoConfig
def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, overwrite_config=None, vpm_device=0, llm_device=0, device_map=None):
    device = torch.device(f'cuda:{vpm_device}')
    if vpm_device != llm_device: # Load onto CPU first
        # device = 'cpu'
        device = vpm_device
    kwargs = {"device_map": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        # kwargs["torch_dtype"] = torch.float16
        kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    if overwrite_config is not None:
        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 32768

    if (device_map is None) and (vpm_device != llm_device): # inference on multi-gpus
        from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map, dispatch_model
        device_map = infer_auto_device_map(model, max_memory={0: "80GB", 1: "80GB"},
            no_split_module_classes=['vision_model', 'LlamaDecoderLayer'])
        device_map["model.vision_tower"] = vpm_device # Vision
        device_map["model.vision_resampler"] = vpm_device
        device_map["model.cubing"] = vpm_device
        device_map["model.embed_tokens"] = vpm_device
        device_map["lm_head"] = vpm_device
        device_map["model.layers"] = llm_device # LLM
        for _ in range(len(model.model.layers)):
            device_map[f"model.layers.{_}"] = llm_device
            device_map[f"model.layers.{_}.self_attn"] = llm_device
            device_map[f"model.layers.{_}.self_attn.q_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.k_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.v_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.o_proj"] = llm_device
            device_map[f"model.layers.{_}.self_attn.rotary_emb"] = llm_device
            device_map[f"model.layers.{_}.mlp"] = llm_device
            device_map[f"model.layers.{_}.mlp.gate_proj"] = llm_device
            device_map[f"model.layers.{_}.mlp.up_proj"] = llm_device
            device_map[f"model.layers.{_}.mlp.down_proj"] = llm_device
            device_map[f"model.layers.{_}.input_layernorm"] = llm_device
            device_map[f"model.layers.{_}.post_attention_layernorm"] = llm_device
            skip_keys = ['inputs_embeds', 'position_ids', 'cache_position', 'attention_mask']
        device_map["model.norm"] = llm_device
        device_map["model.rotary_emb"] = llm_device

        model = dispatch_model(model, device_map=device_map, skip_keys=skip_keys)
    model.eval()

    return tokenizer, model, image_processor, context_len
