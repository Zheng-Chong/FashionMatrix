import math
import os
from datetime import datetime

import gradio as gr

from roles.assistant import FashionAssistant
from utils import move_to_cache

VERSION_DETAILS = """
> Since the Label version of Fashion Matrix has less reliance on LLMs and thus has better stability, we release that version first.
> Dialogue and Instruction versions of Fashion Matrix is coming soon.

Version: *v2.0.0 (Label)*
+ Streamlined supporting models, using less models and GPU memory.
+ Support original image resolution retention (up to 1024x1024).

Version: *v1.1.0 (Label)*
+ Add a new task <u>AI model</u>, which can replace the model while keeping the pose and outfits.
+ Add <u>NSFW (Not Safe For Work) detection</u> to avoid inappropriate using.

Version: *v1.0 (Label)*
+ Basic functions: replace, remove, add, and recolor.
"""

IMAGE_UPLOAD = """NOTE: 
Upload or select a photo from Examples to start editing. 
Editable parts will be automatically detected. """

OUTPAINT_GUIDE = """
<center>↑↑↑</center>
NOTE: 
Click the ✏️ in the upper right corner, zoom the picture with the mouse wheel and click to adjust the cropping frame, and finally click the blank area to confirm the area."""

FA = FashionAssistant()
cache_folder = './cache/v2-0'  # avoid . in the path
part_options = ['hair', 'scarf', "gloves", "sunglasses", 'top', 'bottoms', 'coat', 'jacket', 'dress',
                'neckline', 'sleeves', 'logo', "socks", "shoes", 'necklace', 'watch', 'bracelet', 'belt']

height = 800
width_0, width_1 = 400, 800
scale_0, scale_1 = width_0 // math.gcd(width_0, width_1), width_1 // math.gcd(width_0, width_1)


def submit_function(task, part_reform, part_add, part_input, prompt, src_img):
    global FA
    if task is None:
        raise gr.Error("Please select a task!")

    if task in ['AI model']:
        kwargs = {'task': {'category': 'recolor', 'origin': 'bare body', 'target': prompt},
                  'controlnet': ['inpaint', 'openpose', 'lineart'],
                  'controlnet_conditioning_scale': [0.7, 0.5, 0.4],
                  'control_guidance_start': [0.0, 0.0, 0.0],
                  'control_guidance_end': [1.0, 0.8, 0.3]}
    elif task in ['outpaint']:
        kwargs = {'prompt': prompt}
    else:
        part = part_input if part_input is not None else (part_add if task in ['add'] else part_reform)
        if part is None:
            raise gr.Error("Please select a part or input a part!")
        if task in ['remove']:
            prompt = 'bared skin, clothing'
        kwargs = {'task': {'category': task, 'origin': part, 'target': prompt}}
        if task in ['recolor']:
            kwargs.update({'controlnet': ['inpaint', 'lineart'], 'controlnet_conditioning_scale': [0.7, 0.8]})

    # src_img = move_to_cache(src_img, cache_path=cache_folder)
    kwargs.update({'img_path': src_img})
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {task}')
    print(f'  Source: {os.path.basename(src_img)}')

    kwargs.update({'num_inference_steps': 20})

    try:
        result = FA.submit(**kwargs)
    except Exception as e:
        print(e)
        raise gr.Error(f"Error Occurred! Please try again.\n {e}")

    if isinstance(result, list):
        result = result[0]
    return result


def image_query(img_path):
    # output -> [exist_parts, task]
    if img_path is None or img_path == '':
        return [None, gr.update(visible=False, value='')]
    img_path = move_to_cache(img_path, cache_path=cache_folder)
    return [FA.existing_parts(img_path), gr.update(visible=True)]


def options_update(exist_parts):
    # output -> [part_reform, part_add]
    if exist_parts is None:
        return [gr.update(choice=[], value=None)] * 2
    exist_parts = [p for p in exist_parts if p in part_options]
    return [gr.update(choices=exist_parts),
            gr.update(choices=[part for part in part_options if part not in exist_parts])]


def task_change(task):
    # return: [part_reform, part_add, part_input, prompt, submit]
    reform = task in ['recolor', 'replace', 'remove']
    add = task in ['add']
    part_input = task in ['recolor', 'replace', 'remove', 'add']
    prompt = submit = task is not None and task != '' and task not in ['remove']
    return [gr.update(visible=_) for _ in [reform, add, part_input, prompt, submit]]


if __name__ == '__main__':
    with gr.Blocks(theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg)) as demo:
        gr.Markdown("""<div align ="center"><font size="10">💃 Fashion Matrix </font></div>""")
        with gr.Accordion('Version Details (Click to Expand or Hide)') as accordion:
            gr.Markdown(VERSION_DETAILS)
        with gr.Row().style():
            with gr.Column(scale=scale_0, min_width=width_0):
                exist_parts = gr.State(None)
                input_image = gr.Image(interactive=True, label="Input Image", type="filepath")

                # Warning and Notice
                image_upload_warning = gr.Markdown(IMAGE_UPLOAD, visible=True)
                outpaint_guide = gr.Markdown(OUTPAINT_GUIDE, visible=False)
                # Task
                task = gr.Radio(['replace', 'remove', 'add', 'recolor', 'AI model', 'outpaint'], label="Task",
                                info="Select the task you want to perform on the uploaded image. ", visible=False)
                # Part ( reform, add or input )
                with gr.Group():
                    part_reform = gr.Radio([], label="Part (to be replaced, removed, or recolored)", visible=False,
                                           info="Select the part you want to edit. "
                                                "(This option has lower priority than part(input)")
                    part_add = gr.Radio([], label="Part (to be added)", visible=False,
                                        info="Select the part you want to add. "
                                             "(This option has lower priority than part(input)")
                    part_input = gr.Textbox(label="Part (input)", lines=1, visible=False,
                                            placeholder="Input the part to be edited if it is not listed above. ")
                # Prompt
                prompt = gr.Textbox(label="Text Prompt", lines=2, visible=False,
                                    placeholder="Please describe the target you want to edit, "
                                                "it will directly used as the prompt for the generation.")
                # Buttons
                submit = gr.Button("Start", visible=False)
                accept = gr.Button("Input ← Result", visible=False)

            with gr.Column(scale=scale_1, min_width=width_1):
                # Result
                result_image = gr.Image(interactive=False, label="Result").style(height=height)
                # Photo Examples
                gr.Examples(examples=[os.path.join("examples", f) for f in os.listdir("examples")],
                            inputs=input_image, label="Photo Examples")

            # Input Image Update:  WARNING(visible)  -> Exist Parts -> Task(visible & clear) -> Label(options & clear)
            input_image.change(lambda x: [gr.update(visible=x is None), gr.update(visible=x is not None)], [input_image], [image_upload_warning, task]) \
                .then(image_query, [input_image], [exist_parts, task]) \
                .then(options_update, [exist_parts], [part_reform, part_add])

            # Task Change Update (visible): -> part_reform, part_add, part_input, promp, submit
            task.change(task_change, [task], [part_reform, part_add, part_input, prompt, submit])\
                .then(lambda x: gr.update(visible=x in ['outpaint']), task, outpaint_guide)

            # Result Image Update:  -> accept(visible)
            result_image.change(lambda x: gr.update(visible=x is not None), [result_image], [accept])

            part_reform.select(lambda x: None, [part_reform], [part_input])
            part_add.select(lambda x: None, [part_add], [part_input])

            submit.click(submit_function, [task, part_reform, part_add, part_input, prompt, input_image],
                         [result_image])
            accept.click(lambda x: [x, None], result_image, [input_image, result_image])

    demo.queue().launch(share=True, show_error=True)
