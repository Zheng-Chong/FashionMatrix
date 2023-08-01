import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from graphonomy.dataloaders import custom_transforms as tr
from graphonomy.networks import deeplab_xception_transfer, graph
from matting_anything import networks, utils

"""
DensePose
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
To install detectron2, you can follow:
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' # (add --user if you don't have permission)
"""
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor, CompoundExtractor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

"""
Grounded-SAM (https://github.com/IDEA-Research/Grounded-Segment-Anything)
You can prepare the environment by following:
    python -m pip install -e segment_anything
    python -m pip install -e GroundingDINO
"""
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import load_image
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


class SegmentMattingAnything:
    """
      Codes in this class are modified from:
      1. Matting-Anything (https://github.com/SHI-Labs/Matting-Anything/tree/main)
      2. Grounded-Segment-Anything (https://github.com/IDEA-Research/Grounded-Segment-Anything)
    """
    transform = ResizeLongestSide(1024)

    def __init__(self, model_path="./checkpoints/", device="cuda"):
        mam_checkpoint_path = os.path.join(model_path, "mam_sam_vitb.pth")
        sam_checkpoint_path = os.path.join(model_path, "sam_vit_h_4b8939.pth")
        grounding_dino_checkpoint_path = os.path.join(model_path, "groundingdino_swint_ogc.pth")

        # initialize MAM
        self.mam_model = networks.get_generator_m2m(seg='sam', m2m='sam_decoder_deep').to(device)
        checkpoint = torch.load(mam_checkpoint_path, map_location=device)
        self.mam_model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        self.mam_model = self.mam_model.eval()

        # initialize SAM
        self.sam_model = build_sam(checkpoint=sam_checkpoint_path)
        self.sam_model.to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)

        # initialize GroundingDINO
        self.grounding_dino_model = self.load_grounding_dino(
            model_config_path="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path=grounding_dino_checkpoint_path, device=device)

        self.device = device

    @staticmethod
    def load_grounding_dino(model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        model = self.grounding_dino_model
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def process_mam(self, image_path: str, task_type: str = "text", guidance_mode: str = "alpha",
                    text_prompt: str = 'person', bbox: list = None, point: list = None,
                    iou_threshold: float = 0.5, box_threshold: float = 0.25, text_threshold: float = 0.25):
        assert image_path.endswith('.png'), 'Only support png image'
        assert task_type in ["box", "text", "point"], "task_type must be one of 'box', 'text', 'point'"
        assert guidance_mode in ["alpha", "mask"], "guidance_mode must be one of 'alpha', 'mask'"
        image = Image.open(image_path).convert("RGB")
        original_size = [image.size[1], image.size[0]]
        image = np.array(image)

        if task_type == 'text':
            if text_prompt is None:
                print('Please input non-empty text prompt')
            with torch.no_grad():
                detections, phrases = self.grounding_dino_model.predict_with_caption(
                    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            if len(detections.xyxy) > 1:
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    iou_threshold,
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
            bbox = detections.xyxy[np.argmax(detections.confidence)]
            bbox = self.transform.apply_boxes(bbox, original_size)
            bbox = torch.as_tensor(bbox, dtype=torch.float).to(self.device)

        image = self.transform.apply_image(image)
        image = torch.as_tensor(image).to(self.device)
        image = image.permute(2, 0, 1).contiguous()

        # Normalize
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(self.device)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(self.device)
        image = (image - pixel_mean) / pixel_std

        # Add Padding
        h, w = image.shape[-2:]
        pad_size = image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image = F.pad(image, (0, padw, 0, padh))

        if task_type == 'point':
            assert False, "point mode is not supported yet"

        elif task_type == 'box':
            # Bounding Box
            if bbox is None:
                bbox = [0, 0, original_size[1] - 1, original_size[0] - 1]
            bbox = np.array(bbox)
            bbox = self.transform.apply_boxes(bbox, original_size)
            bbox = torch.as_tensor(bbox, dtype=torch.float).to(self.device)

            sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size,
                      'pad_shape': pad_size}
        elif task_type == 'text':
            sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size,
                      'pad_shape': pad_size}
        else:
            print("task_type:{} error!".format(task_type))

        with torch.no_grad():
            feas, pred, post_mask = self.mam_model.forward_inference(sample)

            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
            alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
            alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
            alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

            alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
            alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
            alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)

            # mask guidance works better on complex scenes with multiple instances
            # alpha guidance works better on simple scene with human instances
            if guidance_mode == 'mask':
                weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
                post_mask[weight_os8 > 0] = alpha_pred_os8[weight_os8 > 0]
                alpha_pred = post_mask.clone().detach()
            else:
                weight_os8 = utils.get_unknown_box_from_mask(post_mask)
                alpha_pred_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
                alpha_pred = alpha_pred_os8.clone().detach()

            weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
            alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]

            weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
            alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

            alpha_pred = alpha_pred[0][0].cpu().numpy()

        # save alpha_pred as PIL image
        alpha_pred = np.uint8(alpha_pred * 255)
        result_path = image_path.replace('.png', f'-MAM_{text_prompt}.png')
        result = Image.fromarray(alpha_pred)
        result.save(result_path)

        return result_path

    def process_sam(self, image_path: str, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        assert image_path.endswith('.png'), 'Only support png image'
        image_source, image = load_image(image_path)
        with torch.no_grad():
            # run grounding dino model
            boxes, phrases = self.get_grounding_output(
                image, text_prompt, box_threshold, text_threshold, device=self.device
            )

        self.sam_predictor.set_image(image_source)
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(
            self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        image_mask = masks[0][0].cpu().numpy()
        result_path = image_path.replace('.png', f'-SAM_{text_prompt}.png')
        image_mask_pil = Image.fromarray(image_mask)
        image_mask_pil.save(result_path)
        return result_path


class DensePose:
    """
    DensePose used in this project is from Detectron2 (https://github.com/facebookresearch/detectron2).
    These codes are modified from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose.
    The checkpoint is downloaded from https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo.

    We use the model R_50_FPN_s1x with id 165712039, but other models should also work.
    The config file is downloaded from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs.
    Noted that the config file should match the model checkpoint and Base-DensePose-RCNN-FPN.yaml is also needed.
    """

    def __init__(self, model_path="./checkpoints", device="cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model_path = os.path.join(model_path, 'model_final_162be9.pkl')
        self.visualizations = ["dp_segm"]
        self.VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer}
        self.min_score = 0.8

        self.cfg = self.setup_config()
        self.predictor = DefaultPredictor(self.cfg)

    def setup_config(self):
        opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(self.min_score)]
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    @staticmethod
    def _get_input_file_list(input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [os.path.join(input_spec, fname) for fname in os.listdir(input_spec)
                         if os.path.isfile(os.path.join(input_spec, fname))]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def create_context(self, cfg, output_path):
        vis_specs = self.visualizations
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = texture_atlases_dict = None
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
                alpha=1.0
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": output_path,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(self, context, entry, outputs):
        extractor = context["extractor"]
        visualizer = context["visualizer"]

        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        image = np.zeros_like(image)
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        cv2.imwrite(context["out_fname"].replace(".png", "_rgb.png"), image_vis)

        H, W, _ = entry["image"].shape
        result = np.zeros((H, W), dtype=np.uint8)

        data, box = data[0]
        x, y, w, h = [int(_) for _ in box[0].cpu().numpy()]
        i_array = data[0].labels[None].cpu().numpy()[0]
        result[y:y + h, x:x + w] = i_array
        result = Image.fromarray(result)
        result.save(context["out_fname"])

    def process(self, image_path: str) -> str:
        output_path = image_path.replace(".png", "-dense.png")
        if not os.path.exists(output_path):
            file_list = self._get_input_file_list(image_path)
            assert len(file_list), "No input images found!"
            context = self.create_context(self.cfg, output_path)
            for file_name in file_list:
                img = read_image(file_name, format="BGR")  # predictor expects BGR image.
                with torch.no_grad():
                    outputs = self.predictor(img)["instances"]
                    self.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        return output_path


class Graphonomy:
    """
    Graphonomy: Universal Human Parsing via Graph Transfer Learning
    These codes and codes in /graphonomy are modified from https://github.com/Gaoyiminggithub/Graphonomy.
    Checkpoints file (trained on CIHP dataset) can be downloaded from the same link above.
    """

    def __init__(self, model_name="./checkpoints/inference.pth", device="cuda"):
        self.device = device
        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                          hidden_layers=128,
                                                                                          source_classes=7, )
        self.net.load_source_model(torch.load(model_name))
        self.net = self.net.to(self.device)
        self.net.eval()

        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        self.adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).to(self.device).transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        self.adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).to(self.device)

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        self.adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).to(self.device)

        self.scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]

    @staticmethod
    def img_transform(img, transform=None):
        sample = {'image': img, 'label': 0}
        sample = transform(sample)
        return sample

    @staticmethod
    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    @staticmethod
    def flip_cihp(tail_list):
        """
        :param tail_list: tail_list size is 1 x n_class x h x w
        :return:
        """
        # tail_list = tail_list[0]
        tail_list_rev = [None] * 20
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
        tail_list_rev[14] = tail_list[15].unsqueeze(0)
        tail_list_rev[15] = tail_list[14].unsqueeze(0)
        tail_list_rev[16] = tail_list[17].unsqueeze(0)
        tail_list_rev[17] = tail_list[16].unsqueeze(0)
        tail_list_rev[18] = tail_list[19].unsqueeze(0)
        tail_list_rev[19] = tail_list[18].unsqueeze(0)
        return torch.cat(tail_list_rev, dim=0)

    def process(self, image_path: str) -> str:
        output_path = image_path.replace(".png", "-graph.png")
        if not os.path.exists(output_path):
            img = Image.open(image_path).convert('RGB')
            origin_size = img.size
            while img.size[1] > 1000:
                w, h = img.size
                img = img.resize((w // 2, h // 2), Image.LANCZOS)

            testloader_list = []
            testloader_flip_list = []
            for pv in self.scale_list:
                composed_transforms_ts = transforms.Compose([
                    tr.Scale_only_img(pv),
                    tr.Normalize_xception_tf_only_img(),
                    tr.ToTensor_only_img()])

                composed_transforms_ts_flip = transforms.Compose([
                    tr.Scale_only_img(pv),
                    tr.HorizontalFlip_only_img(),
                    tr.Normalize_xception_tf_only_img(),
                    tr.ToTensor_only_img()])

                testloader_list.append(self.img_transform(img, composed_transforms_ts))
                testloader_flip_list.append(self.img_transform(img, composed_transforms_ts_flip))

            for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
                inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
                inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
                inputs = inputs.unsqueeze(0)
                inputs_f = inputs_f.unsqueeze(0)
                inputs = torch.cat((inputs, inputs_f), dim=0)
                if iii == 0:
                    _, _, h, w = inputs.size()
                # Forward pass of the mini-batch
                inputs = Variable(inputs, requires_grad=False)

                with torch.no_grad():
                    inputs = inputs.to(self.device)
                    # outputs = net.forward(inputs)
                    outputs = self.net.forward(inputs, self.adj1_test, self.adj3_test, self.adj2_test)
                    outputs = (outputs[0] + self.flip(self.flip_cihp(outputs[1]), dim=-1)) / 2
                    outputs = outputs.unsqueeze(0)

                    if iii > 0:
                        outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                        outputs_final = outputs_final + outputs
                    else:
                        outputs_final = outputs.clone()

            predictions = torch.max(outputs_final, 1)[1]
            results = predictions.cpu().numpy()

            gray = Image.fromarray(np.uint8(results[0]))
            gray = gray.resize(origin_size, Image.NEAREST)
            gray.save(output_path)
        return output_path

