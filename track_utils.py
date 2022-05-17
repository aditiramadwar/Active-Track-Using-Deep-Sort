import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, './yolov5')
import os
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def detect(device, half, dt, model, save_crop, outputs, deepsort_list, names, seen, opt, drone=None, img = None):
    w, h = 360, 240
    im = cv2.resize(img, (640,480))
    im0 = im.copy()
    t1 = time_sync()
    im = torch.from_numpy(im.transpose((2,0,1))).to(device)

    im = im.half() if half else im.float() 
    im /= 255.0 
    if len(im.shape) == 3:
        im = im[None]  
    t2 = time_sync()
    dt[0] += t2 - t1
    pred = model(im, augment=opt.augment)
    t3 = time_sync()
    dt[1] += t3 - t2
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
    dt[2] += time_sync() - t3

    area = []
    centers = []
    boxes = []
  
    label=''
    for i, det in enumerate(pred):  
        seen += 1
        annotator = Annotator(im0, line_width=2, pil=not ascii)
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            t4 = time_sync()
            outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4

            if len(outputs[i]) > 0:
                for j, (output) in enumerate(outputs[i]):

                    bboxes = output[0:4]
                    x_min_norm = output[0]/640
                    x_max_norm = output[2]/640
                    y_min_norm = output[1]/480
                    y_max_norm = output[3]/480
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    x = output[0]
                    y = output[1]
                    cx = x + w // 2
                    cy = y + h // 2
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    ## ID lock on only 1st ID
                    if id=="1":
                        centers=[cx/640,cy/480]
                        boxes=[x_min_norm, y_min_norm, x_max_norm, y_max_norm]
                        area=bbox_w*bbox_h

                    c = int(cls)  # integer class
                    label = f'{id:0.0f} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
 
        else:
            deepsort_list[i].increment_ages()
            LOGGER.info('No detections')

        im0 = annotator.result()
        return im0, [centers, area, boxes], label
        
def init(opt):
    out, source, yolo_model, deep_sort_model, save_txt, imgsz, evaluate, half, \
        project, exist_ok, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    half &= device.type != 'cpu' 
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  
        os.makedirs(out)  

    if type(yolo_model) is str:  
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  
        exp_name = yolo_model[0].split(".")[0]
    else:  
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok) 
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 

    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride) 

    half &= pt and device.type != 'cpu' 
    if pt:
        model.model.half() if half else model.model.float()

    nr_sources = 1

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    names = model.module.names if hasattr(model, 'module') else model.names    
    time.sleep(3.2)
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    return device, half, dt, model, save_crop, outputs, deepsort_list, names, seen