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

from ...dataset import DATASET_TYPE, DATASET_MODALITY
from .utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .modeling import load_pretrained_model
from .utils import build_multi_choice_prompt, build_qa_cot_prompt, build_mcq_cot_prompt
from .utils import extract_characters_regex


def is_multimodal(name):
    mm_ptr = re.compile(r'.*?\.(jpg|png|jpeg|bmp|tiff|webp|mp4|mkv|avi|mov|flv|wmv|webm|m4v)$')
    return mm_ptr.match(name.lower()) is not None

def is_video(name):
    mm_ptr = re.compile(r'.*?\.(mp4|mkv|avi|mov|flv|wmv|webm|m4v)$')
    return mm_ptr.match(name.lower()) is not None

def is_image(name):
    mm_ptr = re.compile(r'.*?\.(jpg|png|jpeg|bmp|tiff|webp)$')
    return mm_ptr.match(name.lower()) is not None


def parse_dialogue(context_list, roles=['human', 'gpt']):
    """ Parse interleaved context into a multi-turn dialogue,
          where each turn starts with N images/videos followed with M prompt-reponse pairs.
    """
    i = 0
    dialogue, rou = [], []
    while i < len(context_list):
        if i==len(context_list)-1 or \
          not is_multimodal(context_list[i]) and is_multimodal(context_list[i+1]):
            rou.append(context_list[i])
            # parse into role value
            msgs, first_added = [{}], False
            for j,value in enumerate(rou):
                if is_multimodal(value) or not first_added:
                    ro = roles[0]
                    msgs[0]['from'] = ro
                    msgs[0]['value'] = msgs[0].get('value', []) + [value] # all imgs are put at begining
                    if not is_multimodal(value):
                        first_added= True
                else:
                    ro = roles[1]
                    if len(msgs) % 2 == 0:
                        ro = roles[0]
                    msgs.append({'from':ro, 'value': value})
            dialogue.extend(msgs)
            rou = []
        else:
            rou.append(context_list[i])
        i += 1
    return dialogue



from .utils import load_image, load_video
from . import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates['qwen2'] # Set conversation template


from ...smp import *
class Quicksviewer(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    IMAGE_TOKEN_INDEX = -200

    video_nframes = 420
    video_fps = 1


    def split_model(self, model_path):
        import math

        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        # embed_tokens, vision_tower, resampler, cubing at cuda:0
        num_layers = 28 # for 8B model
        num_layers_per_gpu = math.ceil(num_layers / num_gpus)
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 8
        num_layers_per_gpu[-1] -= 2
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"model.layers.{layer_cnt}"] = rank + world_size * i
                layer_cnt += 1
        last_gpu = rank + world_size * (num_gpus - 1)
        device_map["model.vision_tower"] = rank # Vision
        device_map["model.vision_resampler"] = rank
        device_map["model.cubing"] = rank
        device_map["model.embed_tokens"] = rank
        device_map["lm_head"] = rank
        device_map["model.norm"] = last_gpu
        device_map["model.rotary_emb"] = last_gpu
        return device_map

    def __init__(self, model_path="quicksviewer/quicksviewer", **kwargs):
        assert model_path is not None

        overwrite_config = {'_attn_implementation':"flash_attention_2"}
        rank, world_size = get_rank_and_world_size()
        device_map = self.split_model(model_path)

        if device_map is None:
            if auto_split_flag():
                logging.warning('Splitting Vision modules and LLM modules into 2 GPUs, separately.')
                self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                    model_path,
                    overwrite_config=overwrite_config,
                    vpm_device=0,
                    llm_device=1
                )
        else:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path,
                device_map=device_map,
                overwrite_config=overwrite_config,
            )
        self.model.eval()
        self.model.tie_weights()


    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        # message = [dict(type='text', value=prompt)]
        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])

        # if use_mpo_prompt:
        #     message = build_mpo_prompt(message, line, dataset)
        return message




    def generate_inner(self, message, dataset=None):
        """
        msg = [
            dict(type='image', value=IMAGE_URL),
            dict(type='image', value=IMAGE_URL),
            dict(type='text', value='How many apples are there in these images?')
        ]
        """
        msgs = [_['value'] for _ in message]
        msgs = parse_dialogue(msgs)

        for rou in msgs:
            values = rou['value']
            values = values if isinstance(values, list) else [values]
            new_values, timestamps = [], None
            for v in values:
                if is_image(v):
                    new_values.append(load_image(v))
                    modality = 'image'
                elif DATASET_MODALITY(dataset) == 'VIDEO' and is_video(v):
                    frames, timestamps = load_video(v, self.video_nframes, self.video_fps)
                    new_values.append(frames)
                else:
                    new_values.append(v)
            rou['value'] = new_values
            rou['timestamps'] = timestamps


        outputs = self.model.chat(
                image=None,
                msgs=msgs, # [{'from':'human'|'gpt','value':str|tuple, 'timestamps':list}]
                modalities=modality, # ['image', 'video', ..]
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                dtype=torch.float16,
                llm_device=torch.device(f'cuda:0')
                # llm_device=torch.device(f'cuda:0') if args.vpm_device!=args.llm_device else None
            )
        outputs = outputs[0]
        
        # Post-process outputs
        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            outputs = extract_characters_regex(outputs)

        print(outputs)
        return outputs


