# limit the number of cpus used by high performance libraries
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
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
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
    # im = im.T
    # print(im.shape)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # print(im.shape)
    # Inference
    # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
    pred = model(im, augment=opt.augment)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
    dt[2] += time_sync() - t3
    info=0
    area = []
    centers = []
    boxes = []
    max_area = 0
    max_center = 0
    max_box = 0
    label=''
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        seen += 1
        # if webcam:  # nr_sources >= 1
        #     p, im0, _ = im0s[i].copy(), dataset.count
        #     p = Path(p)  # to Path
        #     s += f'{i}: '
        #     txt_file_name = p.name
        #     save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        # else:
        #     p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
        #     p = Path(p)  # to Path
        #     # video file
        #     if source.endswith(VID_FORMATS):
        #         txt_file_name = p.stem
        #         save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        #     # folder with imgs
        #     else:
        #         txt_file_name = p.parent.name  # get folder name containing current img
        #         save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

        # txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
        # s += '%gx%g ' % im.shape[2:]  # print string
        imc = im0.copy() if save_crop else im0  # for save_crop

        annotator = Annotator(im0, line_width=2, pil=not ascii)

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to deepsort
            t4 = time_sync()
            outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4
            
            # draw boxes for visualization
            if len(outputs[i]) > 0:
                for j, (output) in enumerate(outputs[i]):

                    bboxes = output[0:4]
                    # print(bboxes) xmin, ymin, xmax, ymax
                    x_min_norm = output[0]/640
                    x_max_norm = output[2]/640
                    y_min_norm = output[1]/480
                    y_max_norm = output[3]/480

                    x_min_denorm = x_min_norm*240
                    x_max_denorm = x_max_norm*240
                    y_min_denorm = y_min_norm*320
                    y_max_denorm = y_max_norm*320
                    bboxes_denorm=[x_min_denorm, y_min_denorm, x_max_denorm, y_max_denorm]
                    boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])


                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]

                    area.append(bbox_w*bbox_h)
                    x = output[0]
                    y = output[1]
                    cx = x + w // 2
            
                    cy = y + h // 2
                    centers.append([cx/640,cy/480])
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    # if save_txt:
                    #     # to MOT format
                    #     bbox_left = output[0]
                    #     bbox_top = output[1]
                    #     bbox_w = output[2] - output[0]
                    #     bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                        #                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{id:0.0f} {names[c]} {conf:.2f}'
                    # print(bboxes)
                    annotator.box_label(bboxes, label, color=colors(c, True))
                    #     if save_crop:
                    #         txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                    #         save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                idx = np.argmax(area)
                max_area = area[idx]
                max_center = centers[idx]
                max_box = boxes[idx]
                

            # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

        else:
            deepsort_list[i].increment_ages()
            LOGGER.info('No detections')

        # Stream results
        im0 = annotator.result()
        return im0, [max_center, max_area, max_box], label
        cv2.imshow("hi", im0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        # 1 millisecond
        # pError = trackFace(info, 640, pid, pError)
        # Save results (image with detections)
        # if save_vid:
        #     if vid_path[i] != save_path:  # new video
        #         vid_path[i] = save_path
        #         if isinstance(vid_writer[i], cv2.VideoWriter):
        #             vid_writer[i].release()  # release previous video writer
        #         if vid_cap:  # video
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         else:  # stream
        #             fps, w, h = 30, im0.shape[1], im0.shape[0]
        #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #     vid_writer[i].write(im0)

# Print results
# t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
# LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
#     per image at shape {(1, 3, *imgsz)}' % t)
# if save_txt or save_vid:
#     s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
#     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
# if update:
#     strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)

# return(bbox_left,bbox_top,bbox_w,bbox_h)

def init(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    # if show_vid:
    #     show_vid = check_imshow()

    # Dataloader
    # if webcam:
    #     show_vid = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    #     nr_sources = len(dataset)
    #     print(nr_sources)
    #     exit(0)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    #     nr_sources = 1
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # drone.connect()
    # drone.streamon()
    # drone.takeoff()
    # drone.send_rc_control(0, 0, 25, 0)
    
    time.sleep(3.2)
    # drone.send_rc_control(0, 0, 0, 0)
    
    w, h = 360, 240
    
    fbRange = [6200, 6800]
    
    pid = [0.4, 0.4, 0]
    
    pError = 0

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # break
    # cap = cv2.VideoCapture(0)
    # while(True):
    # while(cap.isOpened()):
    #     ret, im = cap.read()

        # im = drone.get_frame_read().frame
        # cv2.imshow("Output", im)
        # cv2.waitKey(1)

        # print(im.shape)
        # break
        # continue
        # im.reshape((480,640,3))
    
    return device, half, dt, model, save_crop, outputs, deepsort_list, names, seen