import os
import sys

from models.ssm import SegmentMattingAnything, DensePose, Graphonomy

sys.path.append("/home/chongzheng_p23/data/FashionMatric/")

import json
import uvicorn
from fastapi import FastAPI, Request
from models.vfm import BLIP, MultiControlNet
from utils import assemble_response, torch_gc
import argparse

parser = argparse.ArgumentParser(description='Start Server')
parser.add_argument('-P', '--port', type=int, default=8123, help='Port')
parser.add_argument('-H', '--host', type=str, default='0.0.0.0', help='Host IP')
parser.add_argument('-C', '--checkpoints', type=str, default="./checkpoints", help='path to folder of checkpoints')

app = FastAPI()


@app.post("/blip")
async def blip(request: Request):
    global model_blip
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    image_path = json_post_list.get('image_path')
    question = json_post_list.get('question')
    answer = model_blip.vqa(image_path, question)

    answer = assemble_response({'response': answer})
    torch_gc(model_blip.device)
    return answer


@app.post("/controlnet")
async def controlnet(request: Request):
    global model_cn
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    # image_path = json_post_list.get('image_path')
    # mask_image_path = json_post_list.get('mask_image_path')
    # prompt = json_post_list.get('prompt')
    # added_prompt = json_post_list.get('added_prompt')
    # negative_prompt = json_post_list.get('negative_prompt')
    kwargs = dict(json_post_list)
    answer = assemble_response({'response': model_cn(**kwargs)})
    torch_gc(model_cn.device)
    return answer


@app.post("/graph")
async def graph(request: Request):
    global model_graph
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    image_path = json_post_list.get('image_path')
    response = model_graph.process(image_path=image_path)
    answer = assemble_response({"response": response})
    torch_gc(model_graph.device)
    return answer


@app.post("/densepose")
async def dense_post(request: Request):
    global model_dense
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    image_path = json_post_list.get('image_path')
    response = model_dense.process(image_path=image_path)
    answer = assemble_response({"response": response})
    torch_gc(model_dense.device)
    return answer


@app.post("/matting")
async def matting(request: Request):
    global model_matting
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    result_path = model_matting.process_mam(**dict(json_post_list))
    answer = assemble_response({'response': result_path})
    torch_gc(model_matting.device)
    return answer


@app.post("/segment")
async def segment(request: Request):
    global model_matting
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    result_path = model_matting.process_sam(**dict(json_post_list))
    answer = assemble_response({'response': result_path})
    torch_gc(model_matting.device)
    return answer


if __name__ == '__main__':
    args = parser.parse_args()

    # Vision Foundation Models
    model_blip = BLIP(model_name=os.path.join(args.checkpoints, "blip-vqa-capfilt-large"), device="cuda")
    model_cn = MultiControlNet(controlnet_folder=os.path.join(args.checkpoints, 'ControlNet'), device="cuda")

    # Human Parsing Models
    model_graph = Graphonomy(model_name=os.path.join(args.checkpoints, 'inference.pth'), device='cuda')
    model_dense = DensePose(model_path=args.checkpoints, device='cuda')
    model_matting = SegmentMattingAnything(model_path=args.checkpoints, device='cuda')

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
