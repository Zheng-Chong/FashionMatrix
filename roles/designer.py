import os.path
import time

import numpy as np
from PIL import Image

from api import VFM_API, CHAT_API
from roles.automasker import AutoMasker
from roles.utils import numpy_all_zero

TASK_SPLIT = """Your task is to split Sentence into operations according to the Semantics.
Here are examples:
________________________________________________________________
"Sentence: 'Put a scarf on the person in the picture, and change the top into long sleeves and make it a v-neck.'
"Operations: ['Put a scarf on the person in the picture', 'Change the top into long sleeves', 'Change the top into a v-neck']
________________________________________________________________
Sentence: 'The user requests to change the bottom into a skirt.'
Operations: ['Change the bottom into a skirt.']
________________________________________________________________
Sentence: '{prompt}'
Operations: """
IMAGE_VQA = """IMPORTANT: Your answer strictly adheres to the information I give, and you don't make up anything that doesn't exist in the conversation
Give Information:
{image_info_str}
Combining the above information into a brief, complete and coherent sentence:"""
ORIGIN_DETAIL = """
Give Description based on the given list of features and the original item:
Example:
_____________________________
item: t-shirt
color: white
pattern: solid
length: short
sleeve length: short sleeve
material: cotton
Description: white short cotton t-shirt with short sleeve and solid pattern
_____________________________
Lets start:
_____________________________
item: {ori_item}
{features}
Description:"""
TARGET_DETAIL = """Give description of the Target item after the task, do not fake the description.
_________________________________________________________
Example:
**********************
original: A white cotton t-shirt
task: change the color of the t-shirt to red.
target: A red cotton t-shirt
**********************
original: A white cotton t-shirt
task: change the t-shirt to a blue cotton pullover.
target: A blue cotton pullover
**********************
original: necklace
task: remove the necklace.
target: 
**********************
original: 
task: add a watch on the wrist.
target: A watch on the wrist
_________________________________________________________
Lets start:

original: {ori_detail}
task: {prompt}
target: """
TASK_CLASSIFY_PREFIX = """There are several Task Categories：
1) recolor: change the color of a certain part or area, such as changing the color of the top to yellow.
2) replace: change the shape and appearance of a certain part or partial area, such as modifying the neckline style or sleeve length.
3) add: Add an accessory or clothing that does not worn before, such as wearing a coat, wearing a watch, etc.
4) remove: Erase a certain accessory or part, such as removing necklaces, bracelets, logo, etc.
Determine which CATEGORY the giving prompt belongs to, and the Original item or part before the task and the Target item or part after the task.
----------------------------------------------------------------------------------------------------------------
Here are examples:
**********************
Prompt: change the t-shirt to a blue cotton pullover.
Task: {'category': 'replace', 'origin': 't-shirt', 'target': 'blue cotton pullover'}
**********************
Prompt: change the color of the top to red.
Task: {'category': 'recolor', 'origin': 'top', 'target': 'top'}
**********************
Prompt: Wear black sunglasses on my face.
Task: {'category': 'add', 'origin': '', 'target': 'black sunglasses'}
**********************
Prompt: remove the logo on the top.
Task: {'category': 'remove', 'origin': 'the logo on the top', 'target': ''}
----------------------------------------------------------------------------------------------------------------
"""
TASK_CLASSIFY = """Prompt: {prompt}
Can you give the Task in the form of a python dictionary, keys include category', 'origin' and 'target':"""

ADDED_PROMPT = ", RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, " \
               "high quality, film grain, Fujifilm XT3"
NEGATIVE_PROMPT = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)" \
                  ", text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid" \
                  ", mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, " \
                  "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, " \
                  "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, " \
                  "fused fingers, too many fingers, long neck, UnrealisticDream"


def mask_image(image: Image, mask: np.ndarray) -> Image:
    image = np.array(image)
    image[mask != 0] = 0
    return Image.fromarray(image)


class FashionDesigner:
    def __init__(self):
        self.chat_api = CHAT_API()
        self.vfm_api = VFM_API()
        self.AM = AutoMasker()

    def existing_parts(self, image_path: str) -> list:
        # 1.graph
        body_parts = ['neck', 'big arms', 'forearms', 'thighs', 'legs', 'feet', 'background', 'face', 'hands']
        graph = self.AM.co_segm(image_path)
        graph_keys = [k for k in list(graph.keys()) if k not in body_parts and not numpy_all_zero(graph[k])]
        # 2.VQA
        vqa_keys = ['necklace', 'logo', 'belt']
        vqa_keys = [k for k in vqa_keys
                    if self.vfm_api.vqa(image_path, f"Is there an {k} on the person in the picture? yes or no")
                    .lower().startswith("yes")]
        return graph_keys + vqa_keys

    def fashion_image_caption(self, image_path: str) -> str:
        questions = {
            "gender": ["Is the person a man or a woman?"],
            "top": ["What is the color of the top?", "Is the top long-sleeve, short-sleeve or sleeveless? ",
                    "What's the category of the top wearing by this person?"],
            "bottoms": ["What is the color of the bottoms?", "What is the length of the bottoms?",
                        "What's the category of the bottoms wearing by this person."],
            "dress": ["What is the color of the dress?", "What is the length of the dress?",
                      "What's the category of the dress wearing by this person."],
            "coat": ["What is the color of the coat?",
                     "What's the category of the coat wearing by this person."],
            "accessories": ["Describe the accessories wearing by this person."],
        }
        for part in ["top", "bottoms", "dress", "coat", "accessories"]:
            if self.vfm_api.vqa(image_path, f"Is the person wearing {part}? yes or no").lower().startswith(
                    "no"):
                questions.pop(part)
        image_info = {}
        for key in questions:
            image_info[key] = ", ".join(
                [self.vfm_api.vqa(image_path, question) for question in questions[key]])
        image_info_str = "\n".join([f"-{key}: {image_info[key]}" for key in image_info])
        format_str = IMAGE_VQA.format(image_info_str=image_info_str)
        response, _ = self.chat_api.chat(format_str, [])
        return response

    @staticmethod
    def resize_padding(image_path, width=768, height=768):
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
        image_.save(image_path)
        return crop_box

    def task_category(self, prompt: str) -> dict:
        format_str = TASK_CLASSIFY_PREFIX + TASK_CLASSIFY.format(prompt=prompt)
        response, _ = self.chat_api.chat(format_str, [])
        # print(response)
        response = response[response.find('{'):response.rfind('}') + 1]
        task = eval(response)
        if task['category'] == 'remove':
            task['target'] = 'Bared Skin, Clothing'
        return task

    def detail_of_origin(self, img_path: str, ori_item: str) -> str:
        format_str = "Dose a {ori_item} has the attribute of {attribute}? Answer 'yes' or 'no':"
        gender = self.vfm_api.vqa(img_path, "Is the person in the photo male or female?")
        questions = ['color', 'pattern', 'length', 'sleeve length', "material"]
        features = ""
        for q in questions:
            response, _ = self.chat_api.chat(format_str.format(ori_item=ori_item, attribute=q), [])
            if response.lower() == 'yes':
                features += f"{q}: {self.vfm_api.vqa(img_path, f'What is the {q} of the{ori_item} worn by the {gender}?')}\n"

        format_str = ORIGIN_DETAIL.format(ori_item=ori_item, features=features)
        description, _ = self.chat_api.chat(format_str, [])
        # description = description[description.lower().find("description:") + len("Description:"):].strip()
        return description

    def detail_of_target(self, ori_detail: str, prompt: str) -> str:
        format_str = TARGET_DETAIL.format(ori_detail=ori_detail, prompt=prompt)
        response, history = self.chat_api.chat(format_str, [])
        return response

    def task_split(self, prompt: str) -> list:
        format_str = TASK_SPLIT.format(prompt=prompt)
        response, _ = self.chat_api.chat(format_str, [])
        response = eval(response[response.find("["):response.rfind("]") + 1].replace("\"\"", '\"'))
        assert isinstance(response, list)
        return response

    def verify_legal(self, task: dict, img_path: str) -> bool:
        if task['category'] == 'add' or task['origin'] in self.AM.co_segm(img_path, task['category']):
            return True
        response = self.vfm_api.vqa(image_path=img_path,
                                    question=f"Is the person wearing {task['origin']}?")
        return 'yes' in response.lower()

    def generate(self, task: dict, img_path: str, quick_mode=True, **kwargs):
        print(f"  Task: {task}")
        # resize and padding the image to 768*768
        crop_box = self.resize_padding(img_path)

        # record time cost for each step
        start_time = time.time()

        # Generate the target item Text Prompt
        prompt = task['target']
        if not quick_mode:
            if task['category'] in ['recolor', 'replace']:
                prompt = self.detail_of_target(self.detail_of_origin(img_path, task['origin']), str(task))
            else:
                prompt = self.detail_of_target(task['origin'] if task['category'] == 'remove' else '', str(task))
        print(f"  Prompt: {prompt}")

        # record time cost for each step
        print(f"  Prompt Generation Time: {time.time() - start_time:.2f}s")
        start_time = time.time()

        # Generate the mask for Editing
        mask = self.AM.auto_mask(img_path, task)
        if numpy_all_zero(mask):
            return f"{task['origin']} is not found in the image, please check it."
        mask_path = img_path.replace(".", "-mask.")
        Image.fromarray(mask.astype(np.uint8) * 255).convert('L').save(mask_path)
        masked_image = mask_image(Image.open(img_path), mask)
        masked_image.save(mask_path.replace(".", "ed_image."))

        # record time cost for each step
        print(f"  Mask Generation Time: {time.time() - start_time:.2f}s")
        start_time = time.time()

        # Assemble kwargs
        kwargs['prompt'] = prompt + ADDED_PROMPT
        kwargs['image_path'] = img_path
        kwargs['mask_path'] = mask_path
        if task['category'] == 'recolor' and 'controlnet' not in kwargs:
            kwargs['controlnet'] = ['lineart']

        # Generate the result
        result = self.vfm_api.controlnet(**kwargs)

        if not result.endswith('NSFW.jpg'):
            result_img = Image.open(result).convert("RGB").crop(crop_box)  # recover the image size
            result_img.save(result)

        # record time cost for each step
        print(f"  VFM Generation Time: {time.time() - start_time:.2f}s")

        print(f"  Result Path: {result}")
        return result

    def run(self, tasks: list[str], img_path: str, quick_mode=True, **kwargs) -> str:
        for task in tasks:
            task = self.task_category(task)  # task classification T={c,t_o,t_e}
            img_path = self.generate(task, img_path, quick_mode=quick_mode, **kwargs)
            if not os.path.exists(img_path):  # error occurs, return the error message
                return img_path
        return img_path
