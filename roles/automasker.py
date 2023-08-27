import os

import numpy as np
from PIL import Image

from api import SSM_API, CHAT_API
from roles.utils import (
    first_digits, resize_padding, max_pooling, mask_by_index, mask_nonzero_rectangle, partial_mask, mask_by_rectangle,
    numpy_or, numpy_diff, image_normalization, resize
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
    bared_body_parts = ['neck', 'big arms', 'forearms', 'thighs', 'legs', 'feet']
    co_segm_parts = ['top', 'coat', 'bottoms', 'dress', 'pants', 'skirt', 'sleeves', 'background',
                     'scarf', 'gloves', 'sunglasses', 'socks', 'shoes', 'hat', 'necklace', 'bag', 'ring', 'watch',
                     'hair', 'face', 'arms', 'legs'
                     ]
    add_parts = ['dress', 'coat', 'jacket', 'necklace', 'scarf', 'gloves', 'sunglasses', 'hat', 'logo', 'sleeves']

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
        "pants": [9],
        "skirt": [12],
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

    def graph(self, image_path: str, mode='numpy'):
        img = Image.open(image_path)
        w, h = img.width, img.height
        # Make sure image is .png and <= 512x512
        image_path_ = image_path[:image_path.rfind('.')] + '-norm.png'
        if not os.path.exists(image_path_):
            if max(img.size) > 512:
                img = resize(img, 512, 'long_edge', Image.LANCZOS)
            # save to -norm.png
            img.save(image_path_, 'PNG')

        # Open or generate graph
        graph_path = image_path_[:image_path_.rfind('.')] + '-graph.png'
        if not os.path.exists(graph_path):
            graph_path = self.hpm_api.graph(image_path_)

        # Up-sample graph to original size
        graph = Image.open(graph_path)
        graph = graph.resize((w, h), Image.NEAREST)

        # Post Processing
        if mode == 'numpy':
            return np.array(graph)
        elif mode == 'image':
            return graph
        elif mode == 'dict':
            graph_dict = self.segm_to_dict(graph)
            return {k: v / 255 for k, v in graph_dict.items()}
        else:
            raise NotImplementedError(f'mode {mode} not implemented for graph!')

    def dense(self, image_path: str, mode='numpy'):
        # Make sure image is .png and <= 512x512
        img = Image.open(image_path)
        w, h = img.width, img.height
        image_path_ = image_path[:image_path.rfind('.')] + '-norm.png'
        if not os.path.exists(image_path_):
            if max(img.size) > 1024:
                img = resize(img, 1024, 'long_edge', Image.LANCZOS)
            # save to -norm.png
            img.save(image_path_, 'PNG')

        # Open or generate dense
        dense_path = image_path_[:image_path_.rfind('.')] + '-dense.png'
        if not os.path.exists(dense_path):
            dense_path = self.hpm_api.dense(image_path_)

        # Up-sample dense to original size
        dense = Image.open(dense_path)
        dense = dense.resize((w, h), Image.NEAREST)

        # Post Processing
        if mode == 'numpy':
            return np.array(dense)
        elif mode == 'image':
            return dense
        elif mode == 'dict':
            dense_dict = self.segm_to_dict(dense, segm_type='dense')
            return {k: v / 255 for k, v in dense_dict.items()}
        else:
            raise NotImplementedError(f'mode {mode} not implemented for dense!')

    def segm_to_dict(self, segm: np.array, segm_type='graph') -> dict:
        assert segm_type in ['graph', 'dense'], "Segm Type Must Be In ['graph', 'dense']"
        segm_dict = {}
        segm_index = self.GRAPH_INDEX_MAP.items() if segm_type == 'graph' else self.DENSE_INDEX_MAP.items()
        for name, index in segm_index:
            segm_dict[name] = mask_by_index(segm, index)
        return segm_dict

    def co_segm_efficient(self, image_path: str, parts: list[str], edge_refine=True, merge=False):
        """
        Only segment the parts needed for more efficient inference.

        :param image_path: Source image.
        :param parts: The parts to be segmented.
        :param edge_refine: if True, refine the edge of the person mask by SAM.
        :param merge: if True, merge all parts into one mask.
        :return: A dict of part-mask pairs or a merged mask.
        """
        co_segm_dict = dict()
        graph_dict = self.graph(image_path, mode='dict')

        # Edge Refine using SAM, it can obtain more clear background mask.
        if edge_refine:
            person_mask = np.array(self.hpm_api.segment(image_path, text_prompt='person')) / 255
            for k in graph_dict:
                graph_dict[k] = graph_dict[k] * person_mask
            graph_dict['background'] = 1 - person_mask

        # Graph Only
        if 'bared body' in parts:
            body = numpy_or([graph_dict[_] for _ in ['hair', 'face', 'neck', 'sunglasses', 'arms', 'legs']])
            clothes = numpy_or(
                [graph_dict[_] for _ in ['top', 'bottoms', 'dress', 'scarf', 'gloves', 'socks', 'shoes']])
            co_segm_dict['bared body'] = numpy_diff([max_pooling(body, 11, 1), clothes])
        if 'head' in parts:
            result = numpy_or([graph_dict[_] for _ in ['hair', 'face', 'neck', 'sunglasses']])
            co_segm_dict['head'] = max_pooling(result, 3, 1)

        # Body Parts
        for part in ['neck', 'face', 'hair', 'background']:
            if part in parts:
                co_segm_dict[part] = graph_dict[part]
        # Clothes & Accessories
        for part in ['top', 'dress', 'pants', 'skirt', 'coat', 'scarf', 'gloves', 'socks', 'shoes', 'sunglasses',
                     'hat']:
            if part in parts:
                co_segm_dict[part] = graph_dict[part]

        dense_dict = self.dense(image_path, mode='dict')
        # Arm & Leg Parts
        for part in ['hands', 'big arms', 'forearms']:
            if part in parts:
                co_segm_dict[part] = max_pooling(dense_dict[part], 7, 1) * graph_dict['arms']
        for part in ['legs', 'thighs', 'feet']:
            if part in parts:
                co_segm_dict[part] = max_pooling(dense_dict[part], 7, 1) * graph_dict['legs']

        # Grounded-SAM to detect parts that are not segmented by Graphonomy and DensePose
        for part in ['necklace', 'watch', 'bag', 'ring', 'sleeves']:
            if part in parts:
                co_segm_dict[part] = self.hpm_api.segment(image_path, text_prompt=part,
                                                          box_threshold=0.5, text_threshold=0.5)
                co_segm_dict[part] = np.array(co_segm_dict[part]) / 255

        # Detect parts not been processed ( Unimplemented )
        rest_parts = set(parts) - set(co_segm_dict.keys())
        if len(rest_parts) > 0:
            raise NotImplementedError(f'part {rest_parts} not implemented')

        # Merge all parts into one mask
        if merge:
            return numpy_or([v for v in co_segm_dict.values()])
        return co_segm_dict

    def predefined_add_mask(self, image_path: str, parts: list[str]):
        """
        Predefined mask for add task.
        """
        add_masks = dict()
        graph_dict = self.graph(image_path, mode='dict')
        dense_dict = self.dense(image_path, mode='dict')
        # dress
        if 'dress' in parts:
            add_masks['dress'] = max_pooling(
                numpy_or([graph_dict['top'], graph_dict['bottoms'], graph_dict['coat']]), 5, 1)
        # coat
        if 'coat' in parts:
            add_masks['coat'] = max_pooling(
                partial_mask(numpy_or([graph_dict['top'], graph_dict['arms'], dense_dict['torso']]),
                             [0, 1], [0.38, 0.60], ['False', 'True']), 13, 1)
        # jacket
        if 'jacket' in parts:
            add_masks['jacket'] = max_pooling(
                partial_mask(numpy_or([graph_dict['top'], graph_dict['arms'], dense_dict['torso']]),
                             [0, 1], [0.38, 0.60], ['False', 'True']), 7, 1)
        # necklace
        if 'necklace' in parts:
            add_masks['necklace'] = partial_mask(
                numpy_diff([max_pooling(graph_dict['neck'], 5, 1), graph_dict['face'], graph_dict['top']])
                , [0.1, 1])
        # scarf
        if 'scarf' in parts:
            add_masks['scarf'] = partial_mask(
                numpy_diff([max_pooling(graph_dict['neck'], 11, 2), graph_dict['face'], graph_dict['hair']])
                , [0.1, 1])
        # gloves
        if 'gloves' in parts:
            add_masks['gloves'] = max_pooling(dense_dict['hands'], 9, 1)
        # sunglasses
        if 'sunglasses' in parts:
            add_masks['sunglasses'] = max_pooling(partial_mask(graph_dict['face'], [0.22, 0.58]), 3, 1)
        # hat
        if 'hat' in parts:
            y_0, y_1, x_0, x_1 = mask_nonzero_rectangle(graph_dict['face'])
            y_0, y_1 = max(2 * y_0 - y_1, 0), y_0
            add_masks['hat'] = mask_by_rectangle(np.zeros_like(graph_dict['face']), (y_0, y_1, x_0, x_1))
        # logo
        if 'logo' in parts:
            add_masks['logo'] = partial_mask(dense_dict['torso'] * numpy_or([graph_dict['top'], graph_dict['dress']]),
                                             [0.3, 0.8], [0.25, 0.75])
        # sleeves
        if 'sleeves' in parts:
            add_masks['sleeves'] = numpy_diff([max_pooling(graph_dict['arms'], 11, 1), dense_dict['hands']])

        # Detect parts not been processed ( Unimplemented )
        rest_parts = set(parts) - set(add_masks.keys())
        if len(rest_parts) > 0:
            raise NotImplementedError(f'part {rest_parts} not implemented')

        return add_masks

    def part_classify(self, part: str, options: list[str]):
        item_parts_str = '\n'.join([f'{i + 1}.{p}' for i, p in options])
        format_str = PART_CLASSIFY.format(part=part, item_parts_str=item_parts_str)
        response, _ = self.chat_api.chat(format_str, [])
        if response.lower().startswith('yes'):
            index = int(first_digits(response))
            part = options[index - 1]
            return part
        return None

    def parts_covered(self, task: dict):
        # Possible affected areas
        join_word = {'replace': 'with', 'add': '', 'remove': '', 'recolor': 'to'}[task['category']]
        task_str = f"{task['category']} {task['origin']} {join_word} {task['target']}"
        format_str = COVER_BODY.format(task=task_str, bared_body_parts=self.bared_body_parts)
        response, _ = self.chat_api.chat(format_str, [])
        covered_parts = eval(response[response.find('['):response.find(']') + 1])
        # print("  Covered body parts:", covered_parts)
        return covered_parts

    def __call__(self, image_path: str, task: dict):
        category, origin, target = task.values()
        # Add Task
        if category in ['add']:
            part = self.part_classify(origin, self.add_parts)
            if part:  # Find predefined part
                return self.predefined_add_mask(image_path=image_path, parts=[part])[part]
            else:
                return None
        # recolor, remove, replace Task
        elif category in ['recolor', 'remove', 'replace']:
            part = self.part_classify(origin, self.co_segm_parts)
            if part:  # Find predefined part
                mask = self.co_segm_efficient(image_path=image_path, parts=[part])[part]
            else:  # SAM to detect
                mask = self.hpm_api.segment(image_path=image_path, text_prompt=origin)
                mask = Image.open(mask).convert('L')
                mask = np.array(mask) / 255
            # Max Poling
            if category in ['recolor', 'remove']:
                return max_pooling(mask, 7, 1)
            elif category in ['replace']:
                return max_pooling(
                    numpy_or([mask, self.co_segm_efficient(image_path, parts=self.parts_covered(task))]),
                    9, 1)
        else:
            raise NotImplementedError(f"Task category '{category}' not implemented")

    def clothing_agnostic(self, image):
        """
            Clothing Agnostic Segmentation for Top Warping (HR-VTON Try-on condition generator)
            :return: PIL RGB Image of Clothing Agnostic Segmentation
        """
        graph_np, dense_np = self.graph(image), self.dense(image)
        graph, dense, co_segm = self.segm_to_dict(graph_np), self.segm_to_dict(dense_np, segm_type='dense'), {}
        hands = max_pooling(dense['hands'], 9, 1) * graph['arms']
        left_hand, right_hand = np.zeros_like(hands), np.zeros_like(hands)
        left_hand[graph_np == 14] = 255
        right_hand[graph_np == 15] = 255
        left_hand = left_hand * hands
        right_hand = right_hand * hands

        remove_items = [5, 7, 10, 11, 14, 15]  # ['top', 'coat', 'neck', 'scarf', 'left hand', 'right hand']
        for _ in remove_items:
            graph_np[graph_np == _] = 0
        graph_np[left_hand == 255] = 14
        graph_np[right_hand == 255] = 15

        return Image.fromarray(graph_np.astype(np.uint8), 'P')
