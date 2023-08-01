import os

import numpy as np
from PIL import Image

from api import SSM_API, CHAT_API
from roles.utils import (
    first_digits, resize_padding, max_pooling, mask_by_index, mask_nonzero_rectangle, partial_mask, mask_by_rectangle,
    numpy_or, numpy_diff, image_normalization
)

PART_CLASSIFY = """Does '{part}' belongs to the following categories?
{item_parts_str}
Answer 'yes' or 'no' and then give the index number of the category if yes in the list: """

COVER_BODY = """After replacing or adding fashion item, some bared body parts may be covered.
Example:
***********************************************
Optional body parts: ['neck', 'big arms', 'forearms', 'thighs', 'legs']
Task: Replace the t-shirt with a black sweater
Thought: Sweater usually has Longer sleeve and turtle neckline compared with t-shirt, it may cover the bared neck, big arms, forearms.
Covered body parts: ['neck', 'big arms', 'forearms']
***********************************************
Optional body parts: ['neck', 'big arms', 'forearms', 'thighs', 'legs']
Task: Add a yellow hat
Thought: Hat will not cover any part above.
Covered body parts: []
***********************************************
Optional body parts: ['neck', 'big arms', 'forearms', 'thighs']
Task: Add a watch
Thought: The watch is worn on the bared forearms.
Covered body parts: ['forearms']
***********************************************
Optional body parts: ['neck', 'forearms', 'thighs', 'legs']
Task: Replace shorts with a pair of pants
Thought: Pants are longer than shorts and cover the bared legs, thighs.
Covered body parts: ['legs', 'thighs']
_______________________________________________________
Let's start!
Optional body parts: {bared_body_parts}
Task: {task}
Thought and answer the Covered body parts: """


class AutoMasker:
    GRAPH_INDEX_MAP = {
        "background": [0],
        "hat": [1],
        "hair": [2],
        "face": [13],
        "neck": [10],
        "scarf": [11],
        "gloves": [3],
        "sunglasses": [4],
        "top": [5],
        "dress": [6],
        "coat": [7],
        "socks": [8],
        "bottoms": [9, 12],
        # "skirt": [12],
        "arms": [14, 15],
        "legs": [16, 17],
        "shoes": [18, 19],
    }
    DENSE_INDEX_MAP = {
        "background": [0],
        "torso": [1, 2],
        "right hand": [3],
        "left hand": [4],
        "right foot": [5],
        "left foot": [6],
        "right thigh": [7, 9],
        "left thigh": [8, 10],
        "right leg": [11, 13],
        "left leg": [12, 14],
        "left big arm": [15, 17],
        "right big arm": [16, 18],
        "left forearm": [19, 21],
        "right forearm": [20, 22],
        "face": [23, 24],

        "thighs": [7, 8, 9, 10],
        "legs": [11, 12, 13, 14],
        "hands": [3, 4],
        "feet": [5, 6],
        "big arms": [15, 16, 17, 18],
        "forearms": [19, 20, 21, 22],

    }
    GRAPH_LABEL_COLORS = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85),
                          (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255),
                          (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]

    hpm_api = SSM_API()
    chat_api = CHAT_API()

    def graph_np(self, img_path: str) -> np.ndarray:
        resized_path = resize_padding(img_path, side_length=512)  # Graph has the best support for 512 * 512
        graph_path = self.hpm_api.graph(image_path=resized_path)
        graph = Image.open(graph_path).convert('L')
        graph = np.array(graph)
        return graph

    def graph_rgb(self, img_path: str) -> Image:
        graph = self.graph_np(img_path)
        graph_rgb = np.zeros((graph.shape[0], graph.shape[1], 3))
        for i, rgb in enumerate(self.GRAPH_LABEL_COLORS):
            graph_rgb[graph == i] = rgb
        # print(graph_rgb.shape)
        return Image.fromarray(graph_rgb.astype(np.uint8)).convert('RGB')

    def graph_dict(self, img_path: str) -> dict:
        graph = self.graph_np(img_path)
        graph_dict = self.segm_to_dict(graph)
        return {k: v / 255 for k, v in graph_dict.items()}

    def dense_np(self, img_path: str) -> np.ndarray:
        resized_path = resize_padding(img_path, side_length=512)
        dense_path = self.hpm_api.dense(image_path=resized_path)
        dense = Image.open(dense_path).convert('L')
        dense = np.array(dense)
        return dense

    def dense_rgb(self, img_path: str) -> Image:
        dense_path = self.hpm_api.dense(image_path=img_path)
        dense_path = dense_path.replace('.png', '_rgb.png')
        dense = Image.open(dense_path)
        return dense

    def dense_dict(self, img_path: str) -> dict:
        dense = self.dense_np(img_path)
        dense_dict = self.segm_to_dict(dense, segm_type='dense')
        return {k: v / 255 for k, v in dense_dict.items()}

    def matting(self, img_path: str, prompt: str = 'person', guidance_mode: str = 'alpha') -> Image:
        assert guidance_mode in ['alpha', 'mask']
        matting_path = self.hpm_api.matting(image_path=img_path, text_prompt=prompt, guidance_mode=guidance_mode)
        matting = Image.open(matting_path).convert('L')
        return matting

    def matting_mask(self, image_path, mask, prompt='person'):
        matting = np.array(self.matting(image_path, prompt=prompt))
        return (matting / max(matting.flatten()) * mask).astype(np.uint8)

    def segm_to_dict(self, segm: np.array, segm_type='graph') -> dict:
        assert segm_type in ['graph', 'dense'], "Segm Type Must Be In ['graph', 'dense']"
        segm_dict = {}
        segm_index = self.GRAPH_INDEX_MAP.items() if segm_type == 'graph' else self.DENSE_INDEX_MAP.items()
        for name, index in segm_index:
            segm_dict[name] = mask_by_index(segm, index)
        return segm_dict

    def co_segm(self, img_path: str, task: str = 'replace') -> dict:
        assert os.path.exists(img_path), f'{img_path} does not exist'
        assert task in ['replace', 'recolor', 'remove', 'add'], \
            'task must be in ["replace", "recolor", "remove", "add"]'
        img_path = image_normalization(img_path)
        # matting = np.array(self.matting(img_path, prompt='person', guidance_mode='mask'))

        ori_size = Image.open(img_path).size
        graph = self.graph_dict(img_path)
        dense = self.dense_dict(img_path)

        co_segm = {k: v for k, v in graph.items() if k not in {'arms', 'legs'}}
        # co_segm['background'] = 1 - matting
        co_segm['hands'] = max_pooling(dense['hands'], 5, 1) * graph['arms']
        co_segm['big arms'] = max_pooling(dense['big arms'], 5, 1) * graph['arms']
        co_segm['forearms'] = max_pooling(dense['forearms'], 5, 1) * graph['arms']
        co_segm['thighs'] = max_pooling(dense['thighs'], 5, 1) * graph['legs']
        co_segm['legs'] = max_pooling(dense['legs'], 5, 1) * graph['legs']
        co_segm['feet'] = max_pooling(dense['feet'], 5, 1) * graph['legs']
        co_segm['bare body'] = numpy_or([graph['arms'], graph['legs'], graph['neck'], graph['face'], graph['hair']])

        cover_top = numpy_or([graph['top'], graph['coat'], graph['dress']])
        co_segm['neckline'] = max_pooling(graph['neck'], 11, 2) * cover_top
        co_segm['sleeves'] = numpy_diff([max_pooling(numpy_or([dense['big arms'], dense['forearms']]), 11, 1),
                                         dense['torso']]) * cover_top

        co_segm = {k: v for k, v in co_segm.items() if np.any(v)}  # Remove all zero masks
        if task == 'add':
            if 'hat' not in co_segm:
                y_0, y_1, x_0, x_1 = mask_nonzero_rectangle(graph['face'])
                y_0, y_1 = max(2 * y_0 - y_1, 0), y_0
                co_segm['hat'] = mask_by_rectangle(np.zeros_like(graph['face']), (y_0, y_1, x_0, x_1))
            if 'scarf' not in co_segm and 'neckline' in co_segm:
                co_segm['scarf'] = numpy_or([graph['neck'], co_segm['neckline']])
            if 'gloves' not in co_segm and 'hands' in co_segm:
                co_segm['gloves'] = max_pooling(co_segm['hands'], 5, 1)
            if 'sunglasses' not in co_segm:
                co_segm['sunglasses'] = partial_mask(graph['face'], [0.22, 0.58])
            if 'coat' not in co_segm:
                co_segm['coat'] = partial_mask(numpy_or([graph['top'], graph['arms'], dense['torso']]),
                                               [0, 1], [0.38, 0.60], ['False', 'True'])
            if 'socks' not in co_segm:
                co_segm['socks'] = numpy_diff([partial_mask(graph['legs'], [0.6, 1.0]), graph['shoes']])
            if 'shoes' not in co_segm and 'feet' in co_segm:
                co_segm['shoes'] = max_pooling(co_segm['feet'], 5, 1)
            if 'logo' not in co_segm:
                co_segm['logo'] = partial_mask(dense['torso'] * numpy_or([graph['top'], graph['dress']]), [0.3, 0.8],
                                               [0.25, 0.75])

        # Post Processing
        for s in co_segm:
            # Max Pooling
            co_segm[s] = max_pooling(co_segm[s], 5, 1)
            # Protect Face
            if s not in ['sunglasses', 'hair', 'hat', 'bare body']:
                co_segm[s] = numpy_diff([co_segm[s], max_pooling(graph['face'], 3, 1)])
            # Resize to original size
            co_segm[s] = np.array(Image.fromarray(co_segm[s]).resize(ori_size, Image.NEAREST))

        # # Matting to optimize edge
        # for s in co_segm:
        #     if s != "background":
        #         co_segm[s] = co_segm[s] * matting / np.max(matting.flatten())

        if 'coat' in co_segm:
            co_segm['jacket'] = co_segm['coat']
        return co_segm

    def co_segm_rgb(self, img_path: str) -> Image:
        color_map = {
            'background': (0, 0, 0),

            'hat': (255, 0, 0),
            'sunglasses': (0, 255, 255),
            'face': (0, 0, 255),
            'hair': (0, 255, 0),
            'neck': (0, 0, 64),
            'scarf': (0, 64, 0),

            'dress': (192, 0, 128),
            'top': (255, 128, 0),
            'coat': (192, 128, 0),
            'neckline': (0, 128, 0),
            'sleeves': (128, 128, 0),

            'big arms': (64, 0, 0),
            'forearms': (192, 0, 0),
            'hands': (128, 0, 128),
            'gloves': (0, 128, 128),

            'legs': (0, 128, 128),
            'thighs': (128, 128, 128),
            'bottoms': (64, 0, 128),

            'feet': (0, 64, 128),
            'socks': (128, 64, 0),
            'shoes': (128, 64, 128),
        }
        co_segm = self.co_segm(img_path)
        co_segm_rgb = np.zeros((co_segm['background'].shape[0], co_segm['background'].shape[1], 3))
        for k, rgb in color_map.items():
            if k in co_segm:
                co_segm_rgb[co_segm[k] > 0] = rgb
        return Image.fromarray(co_segm_rgb.astype(np.uint8)).convert('RGB')

    def auto_mask(self, img_path: str, task: dict) -> np.ndarray:
        co_segm = self.co_segm(img_path, task=task['category'])
        co_segm_keys = list(co_segm.keys())

        bared_body_parts = ['neck', 'big arms', 'forearms', 'thighs', 'legs', 'feet']
        bared_body_parts = [k for k in bared_body_parts if k in co_segm_keys]

        item_parts = [k for k in co_segm_keys if k not in bared_body_parts]
        item_parts_str = '\n'.join(f'{i + 1}. {k}' for i, k in enumerate(item_parts))

        # Get Origin Mask
        origin_mask = None
        if task['category'] in ['recolor', 'remove', 'replace']:
            for i in item_parts:
                if i in task['origin'].lower():
                    origin_mask = co_segm[i]
                    print("  Origin Mask (co-segm):", i)
                    break
            if origin_mask is None:
                format_str = PART_CLASSIFY.format(part=task['origin'], item_parts_str=item_parts_str)
                response, _ = self.chat_api.chat(format_str, [])
                if response.lower().startswith('yes'):
                    index = int(first_digits(response))
                    origin_mask = co_segm[item_parts[index - 1]]
                    print("  Origin Mask (co-segm-LLM)::", item_parts[index - 1])
        else:  # add
            origin_mask = np.zeros(co_segm[item_parts[0]].shape)
            part_name = task['origin'].lower() if task['origin'] != '' else task['target'].lower()
            for i in item_parts:
                if i in part_name:
                    origin_mask = co_segm[i]
                    print("  Origin Mask (add):", i)
                    break

        # Grounded-SAM
        if origin_mask is None:
            origin_mask = self.hpm_api.segment(img_path, task['origin'])
            origin_mask = np.array(Image.open(origin_mask).convert('L')) / 255
            print("  Origin Mask (Grounded-SAM):", task['origin'])

        # Postprocess
        if task['category'] == 'remove':
            return max_pooling(origin_mask, 15, 1)
        elif task['category'] == 'recolor':
            return origin_mask
        elif task['category'] in ['replace', 'add']:
            # Possible affected areas
            join_word = {'replace': 'with', 'add': '', 'remove': '', 'recolor': 'to'}[task['category']]
            task_str = f"{task['category']} {task['origin']} {join_word} {task['target']}"
            format_str = COVER_BODY.format(task=task_str, bared_body_parts=bared_body_parts)
            response, _ = self.chat_api.chat(format_str, [])
            covered_parts = eval(response[response.find('['):response.find(']') + 1])
            print("  Covered body parts:", covered_parts)
            for part in covered_parts:
                if part in bared_body_parts:
                    origin_mask = numpy_or([origin_mask, co_segm[part]])
            origin_mask = max_pooling(origin_mask, 9, 1)
            # Protect Face
            if task['origin'] not in ['sunglasses', 'hair', 'hat', 'bare body']:
                origin_mask = numpy_diff([origin_mask, max_pooling(co_segm['face'], 3, 1)])
            return origin_mask
