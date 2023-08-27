import os

import numpy as np
import torch

from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, \
    StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from transformers import BlipProcessor, BlipForQuestionAnswering


def postprocess(output: StableDiffusionPipelineOutput) -> list[Image.Image]:
    images = output.images
    assert isinstance(images[0], Image.Image), "images should be a list of PIL.Image.Image"
    nsfw = output.nsfw_content_detected
    images = [images[i] for i in range(len(images)) if not nsfw[i]]
    return images


def resize_tile_condition_image(input_image: Image, resolution: int, mode="long_side"):
    assert mode in ["long_side", "short_side"], "mode should be long_side or short_side"
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / (min(H, W) if mode == "short_side" else max(H, W))
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


class BLIP:
    def __init__(self, model_name="./checkpoints/blip-vqa-capfilt-large", device='cuda'):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    def vqa(self, image_path: str, question: str) -> str:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer

class InpaintControlNet:
    controlnet_models = {
        'softedge': 'control_v11p_sd15_softedge',
        'openpose': 'control_v11p_sd15_openpose',
        'lineart': 'control_v11p_sd15_lineart'
    }

    def __init__(self,
                 base_model="./checkpoints/realisticVisionV40VAE-inpainting",
                 controlnet_folder: str = "./checkpoints",
                 controlnet: list[str] = None,
                 device='cuda'):

        if controlnet is None:
            controlnet = ['lineart']

        assert all([k in self.controlnet_models for k in controlnet]), \
            f"controlnet should be a subset of {self.controlnet_models.keys()}"

        self.device = device
        self.controlnet = {}
        for k in controlnet:
            model_path = os.path.join(controlnet_folder, self.controlnet_models[k])
            self.controlnet[k] = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model, controlnet=list(self.controlnet.values()), torch_dtype=torch.float16
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def __call__(self, prompt: str, image_path: str, mask_image_path: str,
                 control_image: list[Image.Image] = None,
                 controlnet: list[str] = None, **kwargs) -> list[str]:
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")
        if not isinstance(controlnet, list) and not isinstance(controlnet, str):
            controlnet = ['lineart']
            control_image = [mask_image]
            kwargs['controlnet_conditioning_scale'] = 0.0
        else:
            assert len(controlnet) > 0, "controlnet should be a subset of {self.controlnet.keys()}"
            if isinstance(controlnet, str):
                controlnet = [controlnet]
            if isinstance(control_image, str):
                control_image = [control_image]
            assert len(controlnet) == len(control_image), "'controlnet' and 'control_image' should have the same length"
            assert all([k in self.controlnet for k in controlnet]), \
                f"controlnet should be a subset of {self.controlnet.keys()}"

        if len(controlnet) > 1:
            self.pipe.controlnet = MultiControlNetModel([self.controlnet[k] for k in controlnet]).to(self.device)
        elif len(controlnet) == 1:
            self.pipe.controlnet = self.controlnet[controlnet[0]].to(self.device)

        if 'width' not in kwargs and 'height' not in kwargs:
            kwargs.update({'width': image.width, 'height': image.height})

        with torch.inference_mode():
            results = postprocess(
                self.pipe(prompt=prompt, image=image, mask_image=mask_image,
                          control_image=control_image, **kwargs)
            )
            # save results to the same path of image_path
            result_paths = []
            for i, img in enumerate(results):
                img.save(image_path.replace('.', f'-result-{i}.'))
                result_paths.append(image_path.replace('.', f'-result-{i}.'))
        return result_paths
