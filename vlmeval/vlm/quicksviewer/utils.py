import copy
from typing import Dict, Sequence
import torch
import transformers
import numpy as np
import ast
import math
from PIL import Image
import requests
import io
from io import BytesIO
import cv2
import re
import base64

from . import conversation as conversation_lib
from ..misc import *



CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# DEFAULT_IMAGE_PATCH_TOKEN = "<patch>"
DEFAULT_PATCH_START_TOKEN = "<patch_start>"
DEFAULT_PATCH_END_TOKEN = "<patch_end>"

# DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_START_TOKEN = "<video_start>"
DEFAULT_VIDEO_END_TOKEN = "<video_end>"

# DEFAULT_VIDEO_FRAME_TOKEN = "<frame>"
# DEFAULT_FRAME_START_TOKEN = "<frame_start>"
# DEFAULT_FRAME_END_TOKEN = "<frame_end>"

DEFAULT_THUMBNAIL_START_TOKEN = "<thumbnail_start>"
DEFAULT_THUMBNAIL_END_TOKEN = "<thumbnail_end>"



def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    idxs = sorted(set(idxs))
    return [l[i] for i in idxs], idxs

def split_list(lst, n):
    length = len(lst)
    size = math.ceil(length / n)
    return [lst[i * size:(i + 1) * size] for i in range(n)]



def opencv_extract_frames_fps(vpath, num_frames=None, fps=1.0, start_sec=None, end_sec=None, to_base64=False, to_pilimg=True):
    """ Evenly sampling 'num_frames' frames according to given fps if 'num_frames' is given,
          elsewise sampling all frames according to 'fps'.
    """
    fcap = cv2.VideoCapture(vpath)
    FPS = fcap.get(cv2.CAP_PROP_FPS)
    FCOUNT = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = min(fps, FPS)
    # W, H = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_idx = round(FPS * start_sec) if start_sec else 0
    end_idx = round(FPS * end_sec) if end_sec else int(FCOUNT -1)
    # frame_idx = [i for i in range(start_idx, end_idx+1, round(FPS/fps))]
    frame_idx = [int(i+ii*FPS/fps+FPS/fps/2) for i in range(start_idx, end_idx+1, round(FPS)) for ii in range(round(fps))]
    frame_idx = uniform_sample(frame_idx, num_frames)[0] if num_frames is not None else frame_idx

    # frame = fcap.set(cv2.CAP_PROP_POS_FRAMES, start_sec*FPS)
    frame = fcap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx[0]))
    # writer = cv2.VideoWriter(s_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), FPS, (W, H))
    suc, frame = fcap.read()
    res_frames, timestamps, video_bytes = [], [], []
    ii = frame_idx[0]
    while suc and ii <= frame_idx[-1]:
        if ii in frame_idx:
            # writer.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # add this to make consistent with PIL.Image
            if to_base64:
                imgByteArr = io.BytesIO()
                Image.fromarray(frame).convert('RGB').save(imgByteArr, format='JPEG')
                img_bytes = imgByteArr.getvalue()
                encode_img = base64.b64encode(img_bytes)
                encode_img = str(encode_img, encoding='utf-8')
                video_bytes.append(encode_img)
            if to_pilimg:
                frame = Image.fromarray(frame)
            res_frames.append(frame)
            timestamps.append(round(ii/FPS, 1))
        suc, frame = fcap.read()
        ii += 1
    # writer.release()
    fcap.release()
    if not to_base64:
        return res_frames, timestamps
    else:
        return res_frames, timestamps, video_bytes


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, nframs=None, fps=1):
    video, timestamps = opencv_extract_frames_fps(video_path, nframs, fps, to_pilimg=False)
    video = np.stack(video, axis=0)
    assert video.ndim==4
    # vr = VideoReader(video_path, ctx=cpu(0))
    # fps = round(vr.get_avg_fps()/fps)
    # frame_idx = [i for i in range(0, len(vr), fps)]
    # video = vr.get_batch(frame_idx).asnumpy()
    # timestamps = []
    return video, timestamps


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    # has_image: bool = False,
    no_system_prompt: bool = True,
    build_labels: bool = True,
) -> Dict:
    has_image = DEFAULT_IMAGE_TOKEN in sources[0][0]['value']
    # Note: implemented by yukang2017@, verified by kentang-mit@
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = None
    if build_labels:
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cut = 2 if no_system_prompt else 3
            re_rounds = [conv.sep.join(rounds[:cut])]  # system + user + gpt
            # for conv_idx in range(3, len(rounds), 2):
            for conv_idx in range(cut, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    # import ipdb; ipdb.set_trace()
                    print(f"WARNING: parts!=: {parts}")
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) # Qiji: changed
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    # instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                    instruction_len = len(tokenizer(parts[0]).input_ids) # Qiji: changed

                # # include <|eot_id|> for all rounds
                # round_len += 1
                # instruction_len += 1
                # target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                # cur_len += round_len

                # Qiji: tokenizer_image_token(inp) with llama3 will prepend <|begin_of_text|>, which only appears at first round
                instruction_len = instruction_len-1 if i>0 else instruction_len
                round_len = round_len-1 if i>0 else round_len
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                # cur_len += round_len+1 # +1 for skipping <|eot_id|>
                cur_len += round_len+1 if len(re_rounds)>1 else round_len # +1 for skipping <|eot_id|> at the end of each round

            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    # has_image: bool = False,
    no_system_prompt: bool = True,
    build_labels: bool = True,
) -> Dict:
    has_image = DEFAULT_IMAGE_TOKEN in sources[0][0]['value']
    # Note: implemented by yukang2017@, verified by kentang-mit@
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = None
    if build_labels:
        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cut = 2 if no_system_prompt else 3
            re_rounds = [conv.sep.join(rounds[:cut])]  # system + user + gpt
            # for conv_idx in range(3, len(rounds), 2):
            for conv_idx in range(cut, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    # import ipdb; ipdb.set_trace()
                    print(f"WARNING: parts!=: {parts}")
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) # Qiji: changed
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    # instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                    instruction_len = len(tokenizer(parts[0]).input_ids) # Qiji: changed
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len + len(tokenizer(conv.sep).input_ids) # Qiji: skip "<|im_end|>\n"

            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )



def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len



def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        prompt: str = None,
        refine_prompt: bool = False,
        build_labels: bool = True,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA3:
        return preprocess_llama_3(sources, tokenizer, no_system_prompt=True, build_labels=build_labels)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        return preprocess_qwen_2(sources, tokenizer, no_system_prompt=True, build_labels=build_labels)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers) # TODO for interleave

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal_image(
        sources: Sequence[str],
        data_args,
        pathnums_imgs,
        # layout_format='in_the_front',
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        imgid = 0
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].strip()
                if imgid >= len(pathnums_imgs):
                    break

                new_value = ""
                pre_idx = 0
                while imgid < len(pathnums_imgs):
                     # if mm_layout_format == 'in_the_front':
                    #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    replace_token = DEFAULT_IMAGE_TOKEN
                    if data_args.mm_use_patch_start_end:
                        replace_token = (DEFAULT_PATCH_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_PATCH_END_TOKEN)
                    replace_token = replace_token * pathnums_imgs[imgid]
                    if data_args.mm_use_im_start_end:
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                    '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')


                    cur_idx = sentence['value'].find(DEFAULT_IMAGE_TOKEN, pre_idx)
                    if cur_idx<0:
                        # new_value += sentence['value'][pre_idx:]
                        break
                    new_value += sentence['value'][pre_idx: cur_idx] + replace_token
                    pre_idx = cur_idx + len(DEFAULT_IMAGE_TOKEN)
                    imgid += 1

                new_value += sentence['value'][pre_idx:] 
                sentence["value"] = new_value

    return sources


def preprocess_multimodal_video(
        sources: Sequence[str],
        data_args,
        nframes=64,
        frame_timestamps = [],
        is_cubing=False,
        add_thumbnail=False,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    if len(frame_timestamps)==0:
        frame_timestamps = [""]*nframes
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
            # if DEFAULT_VIDEO_TOKEN in sentence['value']:
                # if data_args.mm_use_frame_start_end:
                #     replace_token = ''.join([ DEFAULT_FRAME_START_TOKEN+str(tmp) + DEFAULT_IMAGE_TOKEN + DEFAULT_FRAME_END_TOKEN for tmp in frame_timestamps])
                # else:
                    # replace_token = ''.join([str(tmp) + DEFAULT_IMAGE_TOKEN for tmp in frame_timestamps])
                replace_token = ''.join([str(tmp) + DEFAULT_IMAGE_TOKEN for tmp in frame_timestamps])

                if add_thumbnail:
                    if data_args.mm_use_thumbnail_start_end:
                        replace_token = DEFAULT_THUMBNAIL_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_THUMBNAIL_END_TOKEN + replace_token
                    else:
                        replace_token = DEFAULT_IMAGE_TOKEN + replace_token

                if data_args.mm_use_video_start_end:
                    replace_token = DEFAULT_VIDEO_START_TOKEN +replace_token + DEFAULT_VIDEO_END_TOKEN
                    
                # if mm_layout_format == 'in_the_front':
                #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Video>' + DEFAULT_IMAGE_TOKEN + '</Video>')
            # replace_token = DEFAULT_IMAGE_TOKEN
            # if data_args.mm_use_im_start_end:
            #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit



def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image



def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches



def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    # patches = divide_to_patches(image_padded, processor.crop_size['height'])
    patches = divide_to_patches(image_padded, processor.crop_size['height'] if hasattr(processor,'crop_size') else processor.size['height']) # support Siglip

    # image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))
    rsize = (processor.size['shortest_edge'],processor.size['shortest_edge']) if 'shortest_edge' in processor.size else (processor.size['width'], processor.size['height'])
    image_original_resize = image.resize(rsize)

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


# def process_images(images, image_processor, model_cfg):
def process_images(images, image_processor, model_cfg, image_aspect_ratio=None, return_pathnums=False):
    if not image_aspect_ratio: image_aspect_ratio=getattr(model_cfg, "image_aspect_ratio", None)
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    # patch_size = model_cfg.patch_size
    new_images = []
    if image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        # return image_processor(images, return_tensors='pt')['pixel_values']
        new_images = [image_processor(img, return_tensors='pt')['pixel_values'] for img in images]
    pathnum_imgs = [len(img) for img in new_images]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    if return_pathnums:
        return new_images, pathnum_imgs
    else:
        return new_images


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    # return width // patch_size, height // patch_size
    return math.ceil(width / patch_size), math.ceil(height / patch_size)

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False



import string
import pandas as pd
def build_multi_choice_prompt(line, dataset=None):
    question = line['question']
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    if hint is not None:
        question = hint + '\n' + question

    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    for key, item in options.items():
        question += f'\n{key}. {item}'
    prompt = question

    if len(options):
        prompt += "\nAnswer with the option's letter from the given choices directly."
    else:
        prompt += '\nAnswer the question directly.'

    return prompt



def build_qa_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding question. The last line of your response should follow this format: "
            "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
            "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
            "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
            "provide your best guess even if unsure. Think step by step logically, considering all "
            "relevant information before answering."
        )
    prompt = prompt + '\n' + cot_prompt

    return prompt

def build_mcq_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding multiple choice question. The last line of your response should follow "
            "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
            "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
            "unsure. Think step by step logically, considering all relevant information before answering."
        )
    prompt = prompt.replace("Answer with the option's letter from the given choices directly.", '').strip()
    prompt = prompt + '\n' + cot_prompt

    return prompt




# Copied from: https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/eval/eval_video_mcqa_videomme.py
def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:"
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]