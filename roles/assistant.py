import os

from api import CHAT_API
from roles.designer import FashionDesigner

SYSTEM_DEFINITION = """You are FashionMatrix. You can answer questions about fashion or give fashion advice.
You have the responsibility to tell the user your ability and identity.
You can also answer questions that unrelated to fashion, but please inform them your identity in advance.
Let's start!
"""

CONTAIN_FASHION = """As a LLM, You have ability to give advice or answer in only text, but can not edit image (e.g., replace, recolor, remove items in the image).
If you have the ability to answer or execute '{prompt}'?",
Think and then Only Answer 'yes' or 'no':
"""

SUMMARY_FASHION = """Dialogue History:
{dialogue_history}

New Input: 
{input}
________________________________________________________________
From the dialogue history, focus on the New Input, which image dose the user want to edit? Please answer the picture path:
In this dialogue history, which Image Path dose the uses want to edit and  What are the editing Requirements?
An requirement should be a single task require to edit a part of the image, such as: Change the color of the pants to blue', 'Swap the dress with a shirt, 'remove the hat', 'Replace the model with a africa lady', 'Add a coat outside the shirt', 'remove the belt' ...
you MUST answer in the format:
```
Image Path: /PATH_TO_THE_IMAGE/image.png
Requirements: ['requirement_1', 'requirement_2', ...]
```
Please Answer:
"""


class FashionAssistant:
    def __init__(self):
        self.llm = CHAT_API()
        self.designer = FashionDesigner()

    def contain_fashion(self, prompt):
        prompt = CONTAIN_FASHION.format(prompt=prompt)
        response, history = self.llm.chat(prompt, [])
        return response.lower().startswith("n")

    def summary_fashion(self, history):
        dialogue_history = "\n".join([f"USER:{i[0]}\nAI:{i[1]}" for i in history[:-1]])
        prompt = SUMMARY_FASHION.format(dialogue_history=dialogue_history, input=history[-1][0])
        response, history = self.llm.chat(prompt, [], temperature=0.01)
        lower_res = response.lower()
        image_path, requirements = lower_res.split("image path:")[1].split("requirements:")
        image_path = image_path.strip()
        requirements = requirements[requirements.find("[") + 1:requirements.find("]")].strip()
        requirements = [_.strip().strip('\'') for _ in requirements.split(",")]

        return image_path, requirements

    def chat(self, prompt, history: list):
        # Upload Image
        if isinstance(prompt, dict):
            prompt = prompt['name']
            print("User Upload Image:", prompt)
            response = self.designer.image_caption(prompt)
            print("  Caption:", response)
            response = f'In the photo you uploaded, I see {response.lower()} What would you like me to do next?'
            format_str = f"Make the following sentence more natural, only return the edited sentence:\n{response}\n"
            response, _ = self.llm.chat(format_str, [])
            return response
        # Chat
        else:
            print("User Input:", prompt)
            history_ = history.copy()
            if self.contain_fashion(prompt):
                image_path, requirements = self.summary_fashion(history_)
                print("  Image Path:", image_path)
                print("  Requirements:", requirements)
                if not os.path.exists(image_path):
                    return "Sorry，I cannot recognize the pictures that need to be edited, please try it out."
                result_path = self.designer.run(requirements, image_path)
                if os.path.exists(result_path):
                    response = (
                        result_path,
                        "I have modified it in accordance with your requirements, please check it out.")
                else:
                    response = result_path
            else:
                history_.insert(0, (SYSTEM_DEFINITION, "OK, FashionMatrix is ready!"))
                response, history_ = self.llm.chat(prompt, history_)
            return response

    def submit(self, **kwargs):
        return self.designer.task_execute(**kwargs)

    def existing_parts(self, img_path: str):
        return self.designer.existing_parts(img_path)
