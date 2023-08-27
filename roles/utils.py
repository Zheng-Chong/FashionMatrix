import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


# if numpy array is all-zero, return bool
def numpy_all_zero(inputs: list[np.ndarray]) -> bool:
    return all([np.sum(i) == 0 for i in inputs])


def numpy_or(inputs: list[np.ndarray]) -> np.ndarray:
    # sum all inputs
    result = np.zeros(inputs[0].shape)
    for i in inputs:
        # if max(i.flatten()) > 1:
        #     i = i / max(i.flatten())
        # assert min(i.flatten()) >= 0 and max(i.flatten()) <= 1, f"Input should be binary mask (0 or 1), " \
        #                                                         f"but got {min(i.flatten())}, {max(i.flatten())}."
        result = result + i
    result[result > 0] = 1
    return result


def numpy_diff(inputs: list[np.ndarray]) -> np.ndarray:
    # sum all inputs
    result = inputs.pop(0)
    assert min(result.flatten()) >= 0 and max(result.flatten()) <= 1, "Input should be binary mask (0 or 1)."
    for i in inputs:
        assert min(i.flatten()) >= 0 and max(i.flatten()) <= 1, "Input should be binary mask (0 or 1)."
        result = result - i
    result[result < 0] = 0
    return result


def convert_jpg_to_png(image_path):
    if image_path.endswith(".png"):
        return image_path
    image = Image.open(image_path)
    image_path = image_path.replace(".jpg", ".png")
    image.save(image_path)
    return image_path


def resize_image(image_path):
    image = Image.open(image_path)
    scale = 512 / max(image.size)
    resize = [round(r * scale) for r in image.size]
    image = image.resize(resize, Image.LANCZOS if image.mode == "RGB" else Image.NEAREST)
    if not image_path.endswith(".png"):
        image_path = image_path[:image_path.rfind(".")] + ".png"
    image.save(image_path)
    return image_path


def first_digits(s):
    s = s.strip()
    if not s:
        return ""
    if s[0].isdigit():
        return s[0] + first_digits(s[1:])
    else:
        return first_digits(s[1:])


def protect_area(src_img, syn_img, mask):
    """
    :param src_img: Original PIL Image
    :param syn_img: Synthesized PIL Image
    :param mask: PIL Gray Image, 0 for syn_img, 1 for src_img
    :return: Mixed PIL Image
    """
    # Convert to tensor
    src = to_tensor(src_img).cuda()
    bgr = to_tensor(syn_img).cuda()
    mask = to_tensor(mask).cuda()
    # Protect area
    result = src * mask + bgr * (1 - mask)
    # src_img, syn_img = np.array(src_img), np.array(syn_img)
    # if not isinstance(mask, np.ndarray):
    #     mask = np.array(mask)
    # mask = mask / max(mask.flatten())
    # # expand mask to 3 channels
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, 3, axis=2)
    # result = src_img * mask + syn_img * (1 - mask)
    return to_pil_image(result.cpu())


def numpy_all_elements(np_array):
    """
    Return set of unrepeated elements in numpy array
    """
    flatten = np_array.flatten()
    list_ = flatten.tolist()
    return set(list_)


def partial_mask(mask, vertical=None, horizontal=None, reverse=None):
    mask_ = mask.copy()
    if horizontal is None:
        horizontal = [0, 1]
    if vertical is None:
        vertical = [0, 1]
    if reverse is None:
        reverse = ["False", "False"]
    assert len(horizontal) == 2 and len(vertical) == 2 and len(reverse) == 2, "Invalid Parameter"

    if vertical[1] - vertical[0] < 1:
        lines = [i for i, line in enumerate(mask_) if np.sum(line) > 0]
        if len(lines) == 0:
            return mask_
        top, bottom = min(lines), max(lines)
        top, bottom = [int(top + (bottom - top) * _) for _ in vertical]
        if reverse[0] == 'True':
            mask_[top:bottom] = 0
        else:
            mask_[:top] = 0
            mask_[bottom:] = 0

    if horizontal[1] - horizontal[0] < 1:
        rows = [i for i, row in enumerate(mask_.T) if np.sum(row) > 0]
        if len(rows) == 0:
            return mask_
        left, right = min(rows), max(rows)
        left, right = [int(left + (right - left) * _) for _ in horizontal]
        if reverse[1] == 'True':
            mask_[:, left:right] = 0
        else:
            mask_[:, :left] = 0
            mask_[:, right:] = 0
    return mask_


def mask_shift(mask, offset):
    assert len(offset) == 2, "Offset Must Be 2-Dimensional (x, y)"
    mask_ = mask.copy()
    mask_ = np.roll(mask_, offset[0], axis=0)
    mask_ = np.roll(mask_, offset[1], axis=1)
    return mask_


def mask_by_rectangle(mask, rectangle):
    assert len(rectangle) == 4, "Rectangle Must Be 4-Dimensional (y_min, y_max, x_min, x_max)"
    mask_ = mask.copy()
    mask_[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]] = 255
    return mask_


def mask_nonzero_rectangle(mask):
    y, x = np.nonzero(mask)
    return np.min(y), np.max(y), np.min(x), np.max(x)


def mask_by_index(mask, index, value=255):
    mask_image = np.zeros_like(mask)
    for i in index:
        mask_image[mask == i] = value
    return mask_image


def max_pooling(array, kernel_size=11, time=2):
    assert kernel_size % 2 == 1, "Kernel Size Must Be Odd Number"
    # assert len(array.shape) == 2, "Array Must Be 2-Dimensional, but {}-Dimensional Given".format(len(array.shape))
    array_ = array.copy().squeeze()
    array_ = torch.from_numpy(array_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    for i in range(time):
        array_ = torch.nn.functional.max_pool2d(array_, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    array_ = array_.squeeze(0).squeeze(0)
    return array_.numpy()


def avg_pooling(array, kernel_size=5, time=1, threshold=0.5):
    assert kernel_size % 2 == 1, "Kernel Size Must Be Odd Number"
    # assert len(array.shape) == 2, "Array Must Be 2-Dimensional, but {}-Dimensional Given".format(len(array.shape))
    array_ = array.copy().squeeze()
    array_ = torch.from_numpy(array_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    for i in range(time):
        array_ = torch.nn.functional.avg_pool2d(array_, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    array_ = array_.squeeze(0).squeeze(0).numpy()
    threshold = 255 * threshold
    array_[array_ > threshold] = 255
    array_[array_ <= threshold] = 0
    return array_


def resize_padding(image_path, side_length=512):
    width = height = side_length
    resize_path = image_path.replace(".", f"-{side_length}.")
    if os.path.exists(resize_path):
        return resize_path
    image = Image.open(image_path)
    w, h = image.size
    if w / width > h / height:
        image = image.resize((width, int(width * h / w)), resample=Image.LANCZOS)
    else:
        image = image.resize((int(height * w / h), height), resample=Image.LANCZOS)
    w_, h_ = image.size
    x_pad, y_pad = (width - image.size[0]) // 2, (height - image.size[1]) // 2
    crop_box = (x_pad, y_pad, x_pad + w_, y_pad + h_)
    image_ = Image.new("RGB", (width, height), (255, 255, 255))
    image_.paste(image, (x_pad, y_pad))
    image_.save(resize_path)
    return resize_path


def image_normalization(img_path: str) -> str:
    """
    Convert image to png and resize to square according to the longer side
    """
    if not img_path.endswith('.png'):
        png_path = img_path[:img_path.rfind('.')] + '.png'
        Image.open(img_path).save(png_path)
        img_path = png_path
    # ori_size = Image.open(img_path).size
    # if ori_size[0] != ori_size[1]:
    #     img_path = resize(img_path, side_length=max(ori_size))
    return img_path


def resize(image: Image.Image, size: int = 512, mode: str = 'long_edge', sample=Image.LANCZOS) -> Image.Image:
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
    w_, h_ = int(w * ratio), int(h * ratio)
    # Make size the nearest multiple of 8
    w_ = w_ - w_ % 8 if w_ % 8 <= 4 else w_ + 8 - w_ % 8
    h_ = h_ - h_ % 8 if h_ % 8 <= 4 else h_ + 8 - h_ % 8
    return image.resize((w_, h_), sample) if w_ != w or h_ != h else image


def padding(image: Image.Image, ratio: float = 1.0, mode='black'):
    """
    :param ratio: H/W
    :param mode: 'black' or 'white'
    """
    assert mode in ['black', 'white'], "mode must be 'black' or 'white'"
    W, H = image.size
    if H / W > ratio:
        width, height = H / ratio, H
    else:
        width, height = W, W * ratio
    width, height = int(width), int(height)
    # Padding to target size
    image_ = Image.new("RGB", (width, height), (0, 0, 0) if mode == 'black' else (255, 255, 255))
    image_.paste(image, (0, 0))
    return image_


def item_box(img: Image.Image):
    # find first and last pixel of non-zero value
    img_ = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    coords = np.column_stack(np.where(gray > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max