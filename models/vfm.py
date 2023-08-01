import os
import time

import numpy as np
import torch
from PIL import Image
from controlnet_aux.processor import Processor
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from transformers import BlipProcessor, BlipForQuestionAnswering


def postprocess(output: StableDiffusionPipelineOutput, prefix: str) -> list[str]:
    images = output.images
    assert isinstance(images[0], Image.Image), "images should be a list of PIL.Image.Image"
    nsfw = output.nsfw_content_detected
    images = [images[i] for i in range(len(images)) if not nsfw[i]]
    result = []
    for i in range(len(images)):
        images[i].save(f"{prefix}-{i}.png")
        result.append(f"{prefix}-{i}.png")
    return result


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


class StableDiffusionInpaint:
    def __init__(self, model_name="./checkpoints/realisticVisionV50_v50VAE-inpainting", device="cuda"):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name,
                                                                   torch_dtype=torch.float16,
                                                                   low_cpu_mem_usage=False,
                                                                   ignore_mismatched_sizes=True).to(device)

    def __call__(self, prompt, image_path, mask_path, **kwargs) -> list[str]:
        """
        :return: List of image paths
        """
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        return postprocess(self.pipe(prompt, image, mask_image, **kwargs), prefix=image_path[:image_path.rfind(".")])


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


class MultiControlNet:
    # More ControlNet models can be find in https://huggingface.co/lllyasviel
    # More preprocessors can be find in https://pypi.org/project/controlnet-aux/0.0.6/
    hf_models = {
        'inpaint': {'name': 'control_v11p_sd15_inpaint', 'processor': None},
        'softedge': {'name': 'control_v11p_sd15_softedge', 'processor': "softedge_pidsafe"},
        'openpose': {'name': 'control_v11p_sd15_openpose', 'processor': "openpose"},
        'lineart': {'name': 'control_v11p_sd15_lineart', 'processor': "lineart_realistic"},
    }

    def __init__(self,
                 base_model="./checkpoints/realisticVisionV50_v50VAE",
                 controlnet_folder: str = "lllyasviel",
                 controlnet: list[str] = None,
                 device='cuda'):
        """
        MultiControlNet can replace ControlNets using during inference to make a variety of effects.
        :param base_model: Path to the RealisticVision model folder
        :param controlnet_folder:  Path to the folder containing the controlnet models
        :param controlnet: List of controlnet names, from ['inpaint', 'softedge', 'openpose', 'lineart']
        :param device: 'cuda' or 'cpu', or torch.device
        """
        if controlnet is None:
            controlnet = ['inpaint', 'softedge', 'lineart', 'openpose']

        assert all([k in self.hf_models for k in controlnet]), \
            f"controlnet should be a subset of {self.hf_models.keys()}"

        self.device = device
        self.controlnet, self.preprocessors = {}, {}
        for k in controlnet:
            model_path = os.path.join(controlnet_folder, self.hf_models[k]['name'])
            self.controlnet[k] = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
            self.preprocessors[k] = Processor(self.hf_models[k]['processor']) if self.hf_models[k][
                'processor'] else None

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model, controlnet=list(self.controlnet.values()), torch_dtype=torch.float16
        ).to(device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def __call__(self, prompt: str, image_path: str, mask_path: str, controlnet: list[str] = None, **kwargs) -> list[
        str]:
        """
        :param controlnet: List of controlnet names ('inpaint' will be used by default)
        :return: List of image paths
        """
        if controlnet is None:
            controlnet = []

        assert all([k in self.controlnet for k in controlnet]), \
            f"Controlnet must be subset of {list(self.controlnet.keys())}."

        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        if 'inpaint' in controlnet:
            controlnet.remove('inpaint')

        # record time cost for each step
        start_time = time.time()

        inpaint_condition = make_inpaint_condition(image, mask_image)
        conditions = [self.preprocessors[k](image, to_pil=True) for k in controlnet]
        for k, c in zip(controlnet, conditions):
            c.save(image_path[:image_path.rfind(".")] + f"-{k}.png")
        controlnet = ['inpaint'] + controlnet

        print(f"Time cost for preprocessing: {time.time() - start_time:.2f}s")
        start_time = time.time()

        # Assemble controlnet conditions
        if len(controlnet) > 1:
            self.pipe.controlnet = MultiControlNetModel([self.controlnet[k] for k in controlnet]).to(self.device)
            images = [image] + conditions
            conditions = [inpaint_condition] + conditions
            if 'controlnet_conditioning_scale' not in kwargs:
                kwargs['controlnet_conditioning_scale'] = [0.75 for _ in controlnet]
        else:
            self.pipe.controlnet = self.controlnet['inpaint'].to(self.device)
            images = image
            conditions = inpaint_condition

        print(f"Time cost for assembling controlnet conditions: {time.time() - start_time:.2f}s")
        start_time = time.time()

        # Inference
        with torch.inference_mode():
            results = postprocess(
                self.pipe(prompt=prompt, image=images, mask_image=mask_image,
                          control_image=conditions, **kwargs),
                prefix=image_path[:image_path.rfind(".")] + f"-{'+'.join(controlnet)}")

        print(f"Time cost for inference: {time.time() - start_time:.2f}s")

        return results
