import os
from datetime import datetime

import torch


def move_to_cache(image_path, cache_path="./cache"):

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    # datetime YYYY-MM-DD
    date = datetime.now().strftime("%Y%m%d")
    # create a folder named with date
    cache_path = os.path.join(cache_path, date)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    # timestamp HHMMSS
    timestamp = datetime.now().strftime("%H%M%S")
    cache_path = os.path.join(cache_path, f'{timestamp}{os.path.splitext(image_path)[1]}')
    os.system(f"mv {image_path} {cache_path}")
    # return the new abspath
    return os.path.abspath(cache_path)


def assemble_response(response: dict):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time}] " + ", ".join([f"{k}: {v}" for k, v in response.items()]))
    response.update({"status": 200, "time": time})
    return response


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

