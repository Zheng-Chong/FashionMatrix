import openai

"""
To deploy Vicuna-13B in OpenAI API, follow the instructions below:
https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
"""


class Vicuna_API:
    def __init__(self, port: int, model="vicuna-13b-v1.3"):
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
