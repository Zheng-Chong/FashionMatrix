import base64
import os
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def resize(image: Image.Image, size: int = 512, mode='long_edge', sample=Image.LANCZOS) -> Image.Image:
    """
    :param sample: Sample method, PIL.Image.LANCZOS or PIL.Image.NEAREST
    :param image: PIL.Image
    :param size: int
    :param mode: str, 'long_edge' or 'short_edge'
    :return: Resized PIL.Image
    """
    assert mode in ['long_edge', 'short_edge'], "mode must be 'long_edge' or 'short_edge'"
    w, h = image.size
    ratio = size / (max(w, h) if mode == 'long_edge' else min(w, h))
    w, h = int(w * ratio), int(h * ratio)
    # Make sure the size is divisible by 8
    w = w - w % 8
    h = h - h % 8
    return image.resize((w, h), sample)


def image_to_list(image: Image.Image):
    return {'image': np.array(image).tolist()}


def list_to_image(image_list: dict):
    image_list = image_list['image']
    arr = np.array(image_list).astype(np.uint8)
    if len(arr.shape) == 3:
        img = Image.fromarray(arr).convert('RGB')
    elif len(arr.shape) == 2:
        img = Image.fromarray(arr).convert('L')
    else:
        raise ValueError(f"Unknown image shape: {arr.shape}")
    return img


def decode_json(data: dict):
    for k, v in data.items():
        if isinstance(v, dict) and 'image' in v:
            data[k] = list_to_image(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict) and 'image' in item:
                    v[i] = list_to_image(item)
    return data


def encode_json(data: dict):
    for k, v in data.items():
        if isinstance(v, Image.Image):
            data[k] = image_to_list(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, Image.Image):
                    v[i] = image_to_list(item)

    # remove None
    data = {k: v for k, v in data.items() if v is not None}
    return data


def encode_pil_to_base64(image: Image.Image):
    image_data = BytesIO()
    image.save(image_data, format='PNG', save_all=True)
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
    return encoded_image


def decode_pil_from_base64(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def move_to_cache(image_path, cache_path="./cache"):
    # create a folder named with date
    cache_path = os.path.join(cache_path, datetime.now().strftime("%Y%m%d"))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # rename as timestamp HHMMSS and move to cache folder
    timestamp = datetime.now().strftime("%H%M%S")
    cache_path = os.path.join(cache_path, f'{timestamp}{os.path.splitext(image_path)[1]}')
    os.system(f"mv {image_path} {cache_path}")

    # return the abspath
    return os.path.abspath(cache_path)


def assemble_response(data: dict):
    # print(data)
    data = encode_json(data)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time}] " + ", ".join([f"{k}: {v if isinstance(v, str) else 'Image'}" for k, v in data.items()]))
    data.update({"status": 200, "time": time})
    return data


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
