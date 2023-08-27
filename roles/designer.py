import time

import numpy as np
from PIL import Image, ImageFilter

from api import VFM_API, CHAT_API, SSM_API
from roles.automasker import AutoMasker
from roles.utils import numpy_all_zero, resize, item_box

IMAGE_VQA = """
{'gender': 'female', 
'top': {'color': 'red', 'pattern': 'none', 'type': 'dress', 'sleeve': 'short sleeve', 'neckline': 'round neck', 'length': 'mini', 'material': 'cotton', 'fit': 'regular fit'},
'bottoms': {'color': 'blue', 'pattern': 'none', 'type': 'shorts', 'length': 'mini', 'material': 'cotton', 'fit': 'regular fit'},
'dress': {'color': 'red', 'pattern': 'none', 'type': 'dress', 'sleeve': 'short sleeve', 'neckline': 'round neck', 'length': 'mini', 'material': 'cotton', 'fit': 'regular fit'},
'coat': {'color': 'red', 'pattern': 'none', 'type': 'coat', 'sleeve': 'long sleeve', 'neckline': 'round neck', 'length': 'mini', 'material': 'cotton', 'fit': 'regular fit'},
'accessories': 'none'}
Combining the above information into a brief, complete and coherent sentence:"""

TASK_CLASSIFY_PREFIX = """There are several Task Categories：
1) recolor: change the color of a certain part or area, such as changing the color of the top to yellow.
2) replace: change the shape and appearance of a certain part or partial area, such as modifying the neckline style or sleeve length.
3) add: Add an accessory or clothing that does not worn before, such as wearing a coat, wearing a watch, etc.
4) remove: Erase a certain accessory or part, such as removing necklaces, bracelets, logo, etc.
5) ai model: Replace the person in the photo but keep the clothing and accessories unchanged.
The Origin item including but not limited to: shirt, jacket, coat, dress, shorts, skirt, pants, sleeves, neckline, shoes, bag, hat, sunglasses, gloves, logo, etc.
Determine which CATEGORY the giving prompt belongs to, and the Original item or part before the task and the Target item or part after the task.

Here are examples:
Prompt: change the t-shirt to a blue cotton pullover.
Task: {'category': 'replace', 'origin': 't-shirt', 'target': 'blue cotton pullover'}

Prompt: change the color of the top to red.
Task: {'category': 'recolor', 'origin': 'top', 'target': 'top'}

Prompt: Wear black sunglasses on my face.
Task: {'category': 'add', 'origin': 'sunglasses', 'target': 'black sunglasses'}

Prompt: remove the logo on the top.
Task: {'category': 'remove', 'origin': 'logo', 'target': 'a top without logo'}

Prompt: Alter the top to have a boat neckline..
Task: {'category': 'replace', 'origin': 'neckline', 'target': 'a top with a boat neckline'}

Prompt: Convert the blazer into a cropped bolero jacket.
Task: {'category': 'replace', 'origin': 'blazer', 'target': 'a cropped bolero jacket'}

Prompt: Give the jacket a faux fur collar for a touch of glamour.
Task: {'category': 'add', 'origin': 'neckline', 'target': 'a faux fur collar'}

"""

TASK_CLASSIFY = """Prompt: {prompt}
Task:"""

ADDED_PROMPT = ", RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, " \
               "high quality, film grain, Fujifilm XT3"

NEGATIVE_PROMPT = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)" \
                  ", text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid" \
                  ", mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, " \
                  "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, " \
                  "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, " \
                  "fused fingers, too many fingers, long neck, UnrealisticDream"


class FashionDesigner:
    chat_api = CHAT_API()
    vfm_api = VFM_API()
    ssm_api = SSM_API()
    AM = AutoMasker()

    def existing_parts(self, image_path: str) -> list:
        # 1.graph
        body_parts = ['neck', 'arms', 'legs', 'feet', 'background', 'face', 'hands']
        graph = self.AM.graph(image_path, 'dict')
        graph_keys = [k for k in list(graph.keys()) if k not in body_parts and not numpy_all_zero(graph[k])]
        # 2.VQA
        vqa_keys = ['necklace', 'logo', 'belt', 'bracelet']
        vqa_keys = [k for k in vqa_keys
                    if self.vfm_api.vqa(image_path, f"Is there an {k} on the person in the picture? yes or no")
                    .lower().startswith("yes")]
        return graph_keys + vqa_keys

    def image_caption(self, image_path: str) -> str:
        questions = {
            "gender": ["Is the person a man or a woman?"],
            "top": ["What is the color of the top?", "What's the category of the top wearing by this person?"],
            "bottoms": ["What is the color of the bottoms?",
                        "What's the category of the bottoms wearing by this person."],
            "dress": ["What is the color of the dress?",
                      "What's the category of the dress wearing by this person."],
            "coat": ["What is the color of the coat?",
                     "What's the category of the coat wearing by this person."],
            "accessories": ["Describe the accessories wearing by this person."],
        }
        for part in ["top", "bottoms", "dress", "coat", "accessories"]:
            if self.vfm_api.vqa(image_path, f"Is the person wearing {part}? yes or no").lower().startswith("no"):
                questions.pop(part)
        image_info = {}
        for key in questions:
            image_info[key] = ", ".join(
                [self.vfm_api.vqa(image_path, question) for question in questions[key]])
        image_info_str = "\n".join([f"-{key}: {image_info[key]}" for key in image_info])
        format_str = IMAGE_VQA.format(image_info_str=image_info_str)
        response, _ = self.chat_api.chat(format_str, [])
        return response

    def task_category(self, prompt: str) -> dict:
        prompt = TASK_CLASSIFY_PREFIX + "Prompt: {prompt}\nTask:".format(prompt=prompt)
        response, _ = self.chat_api.chat(prompt, history=None, temperature=0.01)
        lower_res = response.lower()
        task = eval(lower_res)
        if task['category'] == 'remove':
            task['target'] = 'Bared Skin, Clothing'
        return task

    def task_execute(self, task: dict, image_path: str, mode='quick', **kwargs):
        if task['category'] == 'ai model':
            task['category'] = 'recolor'
            task['origin'] = 'bare bared body'

        # Get target Mask
        mask = self.AM(image_path, task)
        if numpy_all_zero(mask) or mask is None:  # Check mask not None or all zero
            return f"The part '{task['origin']}' is not found in the image, please check it."
        mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')
        mask_path = image_path.replace(".", "-mask.")
        mask.save(mask_path)

        # Get target Prompt
        if mode == 'quick':
            prompt = task['target']
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        print(f"Prompt: {prompt}")

        # Assemble kwargs
        kwargs['prompt'] = prompt + ADDED_PROMPT
        kwargs['image_path'] = image_path
        kwargs['mask_path'] = mask_path
        if 'negative_prompt' not in kwargs:
            kwargs['negative_prompt'] = NEGATIVE_PROMPT
        if task['category'] in ['recolor'] and 'controlnet' not in kwargs:
            kwargs['controlnet'] = ['lineart']
            kwargs['control_image'] = [
                self.vfm_api.lineart(kwargs['image'], False, min(kwargs['image'].width, kwargs['image'].height),
                                     min(kwargs['image'].width, kwargs['image'].height))]

        # Generate the result
        result_paths = self.vfm_api.controlnet(**kwargs)
        if len(result_paths) == 0:
            print("Result: No safe result generated.")
            return './static/images/NSFW.jpg'
        return result_paths[0]

    def __call__(self, image_path: str, tasks: list[str], mode='quick', **kwargs):
        # Pre-Resize image to <= 1024
        img = Image.open(image_path).convert("RGB")
        if max(img.height, img.width) > 1024:
            img = resize(img, 1024, sample=Image.LANCZOS)
            img.save(image_path)

        # Iterate each task
        working_image_path = image_path
        for task in tasks:
            # Task Classification T={c,t_o,t_e}
            T = self.task_category(task)
            print(f"Task: {T}")
            # Execute the task
            working_image_path = self.task_execute(T, working_image_path, mode, **kwargs)
        return working_image_path
