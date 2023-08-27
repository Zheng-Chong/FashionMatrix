import argparse
import json
import uvicorn
from controlnet_aux import LineartDetector
from fastapi import FastAPI, Request
from utils import assemble_response, torch_gc
from PIL import Image
from models.ssm import GroundedSAM, DensePose, Graphonomy
from models.vfm import BLIP, InpaintControlNet

parser = argparse.ArgumentParser(description='Start Server')
parser.add_argument('-P', '--port', type=int, default=8123, help='Port')
parser.add_argument('-H', '--host', type=str, default='0.0.0.0', help='Host IP')
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
    kwargs = dict(json_post_list)
    answer = assemble_response({'response': model_cn(**kwargs)})
    torch_gc(model_cn.device)
    return answer


@app.post("/lineart")
async def lineart(request: Request):
    global model_lineart
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    kwargs = dict(json_post_list)
    kwargs['input_image'] = Image.open(kwargs['input_image']).convert('RGB')
    answer = assemble_response({'response': model_lineart(**kwargs)})
    torch_gc(model_lineart.device)
    return answer


@app.post("/graph")
async def graph(request: Request):
    global model_graph
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    image_path = json_post_list.get('image_path')
    response = model_graph(image_path=image_path)
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
    response = model_dense(image_path=image_path)
    answer = assemble_response({"response": response})
    torch_gc(model_dense.device)
    return answer


@app.post("/segment")
async def segment(request: Request):
    global model_matting
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    result_path = model_matting(**dict(json_post_list))
    answer = assemble_response({'response': result_path})
    torch_gc(model_matting.device)
    return answer


if __name__ == '__main__':
    args = parser.parse_args()

    # Vision Foundation Models
    model_blip = BLIP(device="cuda")
    model_cn = InpaintControlNet(device="cuda")
    model_lineart = LineartDetector.from_pretrained("./checkpoints/Annotators")

    # Human Parsing Models
    model_graph = Graphonomy(device='cuda')
    model_dense = DensePose(device='cuda')
    model_matting = GroundedSAM(device='cuda')

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
