import os

from api import CHAT_API
from roles.designer import FashionDesigner

SYSTEM_DEFINITION = """You are Fashion Assistant. You can answer questions about fashion or give fashion advice.
You have the responsibility to tell the user your ability and identity.
You can also answer questions that unrelated to fashion, but please inform them your identity in advance.
Let's start!
"""

CONTAIN_FASHION = """As a LLM, think if you have the ability to answer or execute '{prompt}'?",
Think and then Only Answer 'yes' or 'no':
"""

SUMMARY_FASHION = """
________________________________________________________________
Dialogue History:
{dialogue_history}

New Input: 
{input}
________________________________________________________________
From the dialogue history, focus on the New Input, which image dose the user want to edit? Please answer the picture path:
In this dialogue history, what are the needs of users who want to edit and the editing Requirements?
Requirements is a list of strings, each string is a single-task requirement, the requirements that have been processed in the dialogue history do not need to be listed.
you MUST answer in the format:
```
Image Path: /PATH_TO_THE_IMAGE/image.png
Requirements: ['requirement_1', 'requirement_2']
```
Let's start:
"""


class FashionAssistant:
    def __init__(self):
        self.chat_api = CHAT_API()
        self.FD = FashionDesigner()

    def contain_fashion(self, prompt):
        prompt = CONTAIN_FASHION.format(prompt=prompt)
        response, history = self.chat_api.chat(prompt, [])
        return response.lower().startswith("n")

    def summary_fashion(self, history):
        dialogue_history = "\n".join([f"USER:{i[0]}\nAI:{i[1]}" for i in history[:-1]])
        prompt = SUMMARY_FASHION.format(dialogue_history=dialogue_history, input=history[-1][0])
        response, history = self.chat_api.chat(prompt, [])
        response = response.strip("\"").strip("\'")
        lower_res = response.lower()
        image_path = response[
                     lower_res.find("image path:") + len("image path:"):lower_res.find("requirements:")].strip()
        requirements = eval(response[lower_res.find("requirements:") + len("requirements:"):].strip())
        return image_path, requirements

    def chat(self, prompt, history: list):
        # Upload Image
        if isinstance(prompt, dict):
            prompt = prompt['name']
            print("User Upload Image:", prompt)
            response = self.FD.fashion_image_caption(prompt)
            print("  Caption:", response)
            response = f'In the photo you uploaded, I see {response.lower()} What would you like me to do next?'
            format_str = f"Make the following sentence more natural, only return the edited sentence:\n{response}\n"
            response, _ = self.chat_api.chat(format_str, [])
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
                result_path = self.FD.run(requirements, image_path)
                if os.path.exists(result_path):
                    response = (
                    result_path, "I have modified it in accordance with your requirements, please check it out.")
                else:
                    response = result_path
            else:
                history_.insert(0, (SYSTEM_DEFINITION, "OK, FashionChatGPT is ready!"))
                response, history_ = self.chat_api.chat(prompt, history_)
            return response

    def submit(self, task: dict, img_path: str, **kwargs):
        return self.FD.generate(task, img_path, **kwargs)

    def existing_parts(self, img_path: str):
        return self.FD.existing_parts(img_path)
