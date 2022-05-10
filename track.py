import argparse
from distutils.log import info
from djitellopy import tello
from tello import trackFace
from track_utils import *



def trackObject(cx, w, pd, pError, drone, fbRange, ar):
    area = ar
    x= cx
    fb = 0
    error = x - w // 2
    speed = pd[0] * error + pd[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    drone.send_rc_control(0, fb, 0, speed)
    return error

def init_drone():
    me = tello.Tello()
    me.connect()
    me.streamon()
    battery = me.get_battery()
    if battery <=30:
        print("Battery level too low")
        exit(0)
    me.takeoff()
    me.send_rc_control(0, 0, 25, 0)
    time.sleep(6.6)
    me.send_rc_control(0, 0, 0, 0)

    return me
if __name__ == '__main__':
    print("parse")
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print("track")
    out = cv2.VideoWriter('results_multi.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (960, 720))
    
    device, half, dt, model, save_crop, outputs, deepsort_list, names, seen = init(opt)
    # exit(0)
    # me = tello.Tello()
    
    # w, h = 360, 240
 
    fbRange = []
    
    pd = [0.4, 0.4, 0]

    pError = 0
    t_w, t_h = 360, 240
    
    first_flag = 1
    # Connect to tello, check battery and take off
    drone = init_drone() 
    with torch.no_grad():
        while(True):

            frame = drone.get_frame_read().frame
            h, w = frame.shape[:2]
            og=frame.copy()
            # Obtain DeepSort detections
            frame, info, label = detect(device, half, dt, model, save_crop, outputs, deepsort_list, names, seen, opt, img = frame) 
            

            bbox = info[2]
            
            if bbox!=0:
                # print("bbox : ",bbox)
                
                ar=info[1]//100

                xmin, xmax = bbox[0], bbox[2]
                ymin, ymax = bbox[1], bbox[3]
                t_xmin, t_xmax = xmin*t_w, xmax*t_w
                ymin, ymax = ymin*h, ymax*h
                xmin, xmax = xmin*w, xmax*w
                cx, cy = xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2
                
                t_cx = t_xmin + (t_xmax-t_xmin)/2
                p1=(int(xmin),int(ymin))
                p2=(int(xmax),int(ymax))

                cv2.rectangle(og, p1, p2, (0,0,255), 2, cv2.LINE_AA)
                tf = 1  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(og, p1, p2, (0,0,255), -1, cv2.LINE_AA)  # filled

                cv2.putText(og, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, (255, 255, 255),
                            thickness=tf, lineType=cv2.LINE_AA)
                cv2.circle(og, (int(cx), int(cy)), 4, (0,0,255), -1)
                print(ar)
                if first_flag==1:
                    fbRange = [ar-200, ar+200]
                    first_flag=0
                else:
                    pError = trackFace(t_cx, t_w, pd, pError, drone, fbRange, ar)
            cv2.imshow('Frame',og)

            out.write(og)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                drone.streamoff()
                drone.land()
                break

        # cap.release()
        cv2.destroyAllWindows()
