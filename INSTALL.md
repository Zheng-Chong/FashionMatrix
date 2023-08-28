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

```

### Visual Foundation Models & Semantic Segmentation Models

```bash
# For most checkpoints, we provide them in HuggingFace, just clone it.
git clone https://huggingface.co/zhengchong/FashionMatrix ./checkpoints

# For ControlNet and BLIP, You need to clone them manually.
cd checkpoints
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
git clone https://huggingface.co/Salesforce/blip-vqa-capfilt-large
```

The checkpoints folder should look like this:
```bash
checkpoints
├── blip-vqa-capfilt-large
├── control_v11p_sd15_lineart
├── realisticVisionV40_v40VAE
├── densepose
    ├── Base-DensePose-RCNN-FPN.yaml
    ├── densepose_rcnn_R_50_FPN_s1x.yaml
    ├── model_final_162be9.pkl
├── graphonomy
    ├── inference.pth
├── grounded-sam
    ├── groundingdino_swint_ogc.pth
    ├── sam_vit_b_01ec64.pth
├── Annotators
    ├── sk_model.pth
    ├── sk_model2.pth
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
