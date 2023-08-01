import math
import os
from datetime import datetime

import gradio as gr

from roles.assistant import FashionAssistant
from utils import move_to_cache

NOTE = """
> Since the Label version of Fashion Matrix has less reliance on LLMs and thus has better stability, we release that version first.
> Dialogue and Instruction versions of Fashion Matrix is coming soon.

Version: *v1.1 (Label)*
+ Add a new task <u>AI model</u>, which can replace the model while keeping the pose and outfits.
+ Add <u>NSFW (Not Safe For Work) detection</u> to avoid inappropriate using.

Version: *v1.0 (Label)*
+ Basic functions: replace, remove, add, and recolor.
"""
FA = FashionAssistant()
cache_folder = './cache/v1-1'  # avoid . in the path
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
    else:
        part = part_input if part_input is not None else (part_add if task in ['add'] else part_reform)
        if part is None:
            raise gr.Error("Please select a part or input a part!")
        if task in ['remove']:
            prompt = 'bared skin, clothing'
        kwargs = {'task': {'category': task, 'origin': part, 'target': prompt}}
        if task in ['recolor']:
            kwargs.update({'controlnet': ['inpaint', 'lineart'], 'controlnet_conditioning_scale': [0.7, 0.8]})

    src_img = move_to_cache(src_img, cache_path=cache_folder)
    kwargs.update({'img_path': src_img})
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {task}')
    print(f'  Source: {os.path.basename(src_img)}')

    kwargs.update({'num_inference_steps': 20})

    try:
        result = FA.submit(**kwargs)
    except Exception as e:
        print(e)
        raise gr.Error("Error Occurred! Please try again.")

    if os.path.isfile(result):
        return result
    else:  # error
        raise gr.Error(result)


def image_change(img_path):
    # return: [warning, task, exist_parts]
    if img_path is None:
        return [gr.update(visible=True),
                gr.update(visible=False, value=None), []]
    else:
        parts = FA.existing_parts(img_path)
        # print(parts)
        return [gr.update(visible=False),
                gr.update(visible=True, value=None), parts]


def task_select(task, exist_parts):
    exist_parts = [p for p in exist_parts if p in part_options]
    # return: [part_reform, part_add, part_input, prompt, submit]
    if task is None or task == '':
        return [gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False)]
    if exist_parts is None:
        raise gr.Error("Please upload an image first!")
    if task == 'add':
        return [gr.update(visible=False),
                gr.update(visible=True, choices=exist_parts),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
    elif task == 'remove':
        clothes = ['top', 'bottoms', 'coat', 'jacket', 'dress', 'neckline', 'hair']
        return [gr.update(visible=True, choices=[part for part in exist_parts if part not in clothes]),
                gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=True)]
    elif task == 'AI model':
        return [gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=True)]
    else:
        return [gr.update(visible=True, choices=exist_parts), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True)]


if __name__ == '__main__':
    with gr.Blocks(theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg)) as demo:
        gr.Markdown("""<div align ="center"><font size="10">💃 Fashion Matrix </font></div>""")
        with gr.Accordion('Version Details (Click to Expand or Hide)') as accordion:
            gr.Markdown(f'{NOTE}')
        with gr.Row().style():
            with gr.Column(scale=scale_0, min_width=width_0):
                exist_parts = gr.State(None)

                input_image = gr.Image(interactive=True, label="Input Image", type="filepath")
                image_upload_warning = gr.Markdown("NOTE: <br> "
                                                   "Upload or select a photo from Examples to start editing. <br>"
                                                   "Editable parts will be automatically detected. "
                                                   , visible=True)

                task = gr.Radio(['replace', 'remove', 'add', 'recolor', 'AI model'], label="Task",
                                info="Select the task you want to perform on the uploaded image. ", visible=False)

                with gr.Group() as options:
                    part_reform = gr.Radio(
                        [],
                        label="Part (to be replaced, removed, or recolored)",
                        info="Select the part you want to edit. (This option has lower priority than part(input)",
                        visible=False
                    )

                    part_add = gr.Radio(
                        [],
                        label="Part (to be added)",
                        info="Select the part you want to add. (This option has lower priority than part(input)",
                        visible=False, interactive=True
                    )

                    part_input = gr.Textbox(label="Part (input)", lines=1, visible=False,
                                            placeholder="Input the part to be edited if it is not listed above. ")

                prompt = gr.Textbox(label="Text Prompt", lines=2, visible=False,
                                    placeholder="Please describe the target you want to edit, "
                                                "it will directly used as the prompt for the generation.")

                submit = gr.Button("Start", visible=False)
                accept = gr.Button("Input ← Result", visible=False)

            with gr.Column(scale=scale_1, min_width=width_1):
                result_image = gr.Image(interactive=False, label="Result").style(height=height)

                gr.Examples(
                    examples=[os.path.join("examples", f) for f in os.listdir("examples")],
                    inputs=input_image,
                    label="Photo Examples",
                )

            input_image.change(image_change, [input_image], [image_upload_warning, task, exist_parts])
            result_image.change(lambda x: gr.update(visible=x is not None), [result_image], [accept])
            task.change(task_select, [task, exist_parts], [part_reform, part_add, part_input, prompt, submit])

            part_reform.select(lambda x: None, [part_reform], [part_input])
            part_add.select(lambda x: None, [part_add], [part_input])

            submit.click(submit_function, [task, part_reform, part_add, part_input, prompt, input_image], [result_image])
            accept.click(lambda x: [x, None], result_image, [input_image, result_image])

    demo.queue().launch(share=True)
