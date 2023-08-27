# Installation

## Environment

### Create a conda environment
```bash
conda create -n fashionmatrix python=3.9
conda activate fashionmatrix
```
### Install Packages
```bash
cd FashionMatrix
pip install -r requirements.txt
```

For Detectron2, Segment Anything and GroundingDINO, install them locally:
```bash
# Detectron2(DensePose), SAM and GroundingDINO
python -m pip install -e detectron2
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```

## Models
```bash
mkdir checkpoints
cd checkpoints
```
### Semantic Segmentation Models
```bash
# Download the pretrained groundingdino-swin-tiny and sam-vit-h models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download MAM models
https://drive.google.com/drive/folders/1Bor2jRE0U-U6PIYaCm6SZY7qu_c1GYfq?usp=sharing

# Download DensePose Model and Config
wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
wget https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml
wget https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml

# Graphonomy Model can be downloaded from:
https://drive.google.com/file/d/1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9/view
```

### Visual Foundation Models
>Realistic Vision V4.0 (V5.0 is a bit unstable in our test) finetuned from SD-v1.5 can be downloaded from [CivitAI](https://civitai.com/models/4201?modelVersionId=125411).
>After downloading the safetensor file, you must convert it to diffusers format following instructions [here](https://github.com/haofanwang/Lora-for-Diffusers#full-models).

```bash
# ControlNet (Make sure you have git-lfs installed (https://git-lfs.com))
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_softedge
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_openpose

# BLIP
git clone https://huggingface.co/Salesforce/blip-vqa-capfilt-large
```

The checkpoints folder should look like this:
```bash
checkpoints
├── blip-vqa-capfilt-large
├── ControlNet
    ├── control_v11p_sd15_inpaint
    ├── control_v11p_sd15_softedge
    ├── control_v11p_sd15_lineart
    ├── control_v11p_sd15_openpose
├── realisticVisionV40_v40VAE
├── Base-DensePose-RCNN-FPN.yaml
├── densepose_rcnn_R_50_FPN_s1x.yaml
├── model_final_162be9.pkl
├── groundingdino_swint_ogc.pth
├── sam_vit_h_4b8939.pth
├── mam_sam_vitb.pth
├── inference.pth
```
### LLM
To deploy Vicuna-13B in OpenAI API, follow this [instructions](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md).

## Get Started
Firstly, deploy the models on a local server:
```bash
# You can change anther host:port, but make sure to change the corresponding address in api.py
CUDA_VISIBLE_DEVICES='YOUR_DEVICE_ID' nohup python -u server.py --host=0.0.0.0 --port=8123 >server.log 2>&1 &
```
It is worth mentioning that all models (except LLM) can be run on a single consumer-grade GPU with 13G+ VRAM.

Then, run the gradio app:
```bash
python app_label.py
```
