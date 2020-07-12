# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import glob
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-d', '--data', type=str, default=f'datasets/wheat/val/', help='dir with imgs')
ap.add_argument('-w', '--weights', type=str, default=f'weights/effdet_wheat-d5.pth', help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=bool, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=bool, default=False)
ap.add_argument('--override', type=bool, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = 5
nms_threshold = 0.33
use_cuda = args.cuda
gpu = 0
use_float16 = False
override_prev_results = 1
project_name = "wheat"
weights_path = args.weights
data_path = args.data

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def evaluate_coco(img_path, model, threshold=0.05):
    kag_res = []
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    imgs_files = [os.path.join(img_path, fn) for fn in os.listdir(img_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for img_path in tqdm(imgs_files):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            kag_res.append(f"{os.path.basename(img_path).replace('.jpg', '')}, {format_prediction_string(rois, scores)}")

    if not len(kag_res):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'/kaggle/working/submission.csv'
    if os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        for line in kag_res:
            f.write(line)
            f.write("\n")


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


if __name__ == '__main__':
    VAL_IMGS = data_path
    # VAL_IMGS = r"C:\Projects\kaggle\wheat\test"
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()

    evaluate_coco(VAL_IMGS, model)
