import openai
import requests


class API:
    def __init__(self, host="0.0.0.0", port=8000):
        self.prefix = "http://{}:{}/".format(host, port)

    def post(self, endpoint, data):
        return requests.post(self.prefix + endpoint, json=data).json()


class VFM_API(API):
    def __init__(self, host='0.0.0.0', port=8123):
        super().__init__(host, port)

    def vqa(self, image_path: str, question: str = None):
        if question is None:
            question = "Describe the image:"
        response = self.post("blip", {'image_path': image_path, 'question': question})
        return response.get('response')

    def controlnet(self, image_path: str, mask_path: str, prompt: str, **kwargs) -> str:
        content = {"prompt": prompt, "image_path": image_path, "mask_path": mask_path}
        content.update(kwargs)
        response = self.post("controlnet", content).get('response')  # return List[str]
        response = response[0] if len(response) > 0 else "./static/images/NSFW.jpg"  # NSFW
        return response


class SSM_API(API):
    def __init__(self, host='0.0.0.0', port=8123):
        super().__init__(host, port)

    def graph(self, image_path: str) -> str:
        response = self.post('graph', {'image_path': image_path})
        return response.get('response')

    def dense(self, image_path: str) -> str:
        response = self.post('densepose', {'image_path': image_path})
        return response.get('response')

    def matting(self, image_path: str, task_type: str = "text", guidance_mode: str = "alpha",
                text_prompt: str = 'person', bbox: list = None, point: list = None,
                iou_threshold: float = 0.5, box_threshold: float = 0.25, text_threshold: float = 0.25) -> str:
        response = self.post('matting', {'image_path': image_path, 'task_type': task_type,
                                         'guidance_mode': guidance_mode, 'text_prompt': text_prompt,
                                         'bbox': bbox, 'point': point, 'iou_threshold': iou_threshold,
                                         'box_threshold': box_threshold, 'text_threshold': text_threshold})
        return response.get('response')

    def segment(self, image_path: str, text_prompt: str = 'person',
                box_threshold: float = 0.3, text_threshold: float = 0.25) -> str:
        response = self.post('segment', {'image_path': image_path, 'text_prompt': text_prompt,
                                         'box_threshold': box_threshold, 'text_threshold': text_threshold})
        return response.get('response')


class CHAT_API:
    def __init__(self, port: int=8001, model="vicuna-13b-v1.3"):
        super().__init__()
        self.model = model
        openai.api_base = f"http://localhost:{port}/v1"
        openai.api_key = "EMPTY"

    def chat(self, prompt, history, temperature=0.1):
        history_ = []
        for u, a in history:
            history_.append({"role": 'user', "content": u if u is not None else ""})
            history_.append({"role": 'assistant', "content": a if a is not None else ""})

        history_.append({"role": "user", "content": prompt})
        completion = openai.ChatCompletion.create(model=self.model, messages=history_, temperature=temperature)
        response = completion.choices[0].message.content
        history_.append({"role": "assistant", "content": response})

        history__ = []
        for i in range(0, len(history_), 2):
            history__.append((history_[i]["content"], history_[i + 1]["content"]))
        return response, history

