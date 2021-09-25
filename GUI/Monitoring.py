from PyQt5.QtWidgets import QFrame
from GUI.Ui_Monitoring import Ui_Monitoring
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
from pathlib import Path
import os.path
import numpy as np
import time

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import os
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
from SQL_Connection.SQLConnection import SQLConnection
import datetime

path_save_violating_vehicle = "violating_vehicle/"

violation_time = {}

SQL = SQLConnection()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

pixels_per_meter = 1
fps = 0

fileName = ""
videoFile = ""

carStartPosition = {}
carCurrentPosition = {}
speed = {}
violating_vehicle = {}
violating_name = {}

def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, speed = {}, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        if not (speed == {}):
            if id in speed:
                cv2.putText(img, str(speed[id]) + "km/h", (x1 + 30, y1 + t_size[1] + 20), cv2.FONT_HERSHEY_PLAIN, 4, [255, 255, 255], 2)
    return img

flag = False

def calculate_speed(startPosition, currentPosition, fps):
    global pixels_per_meter
    xG_start, yG_start = (startPosition[0] + startPosition[2]) / 2, (startPosition[1] + startPosition[3]) / 2
    xG_current, yG_current = (currentPosition[0] + currentPosition[2]) / 2, (currentPosition[1] + currentPosition[3]) / 2
    distance_in_pixels = math.sqrt(math.pow(xG_current - xG_start, 2) + math.pow(yG_current - yG_start, 2))
    distance_in_meters = distance_in_pixels / pixels_per_meter
    speed_in_meter_per_second = distance_in_meters * fps
    speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

    if not speed == {}:
        for id in carCurrentPosition.keys():
            if id in speed:
                if(speed[id] < 30 or speed[id] > 150):
                    if id not in violating_vehicle:
                        violating_vehicle[id] = carCurrentPosition[id]
                        violation_time[id] = datetime.datetime.now()
                    if(speed[id] == 0):
                        violating_name[id] = "Parking"
                    if(speed[id] < 30 and speed[id] > 0):
                        violating_name[id] = "Under Speed"
                    if(speed[id] > 150):
                        violating_name[id] = "Over Speed"
                    if(yG_start > 500 and yG_current < 50):
                        violating_name[id] = "Opposite Lane"

    return speed_in_kilometer_per_hour 

def cleanDictSpeed():
    global speed
    global carStartPosition
    global carCurrentPosition
    speed.clear()
    carStartPosition.clear()
    carCurrentPosition.clear()
    violating_vehicle.clear()

class ThreadMonitoring(QThread): # Using thread for real-time detect and tracking
    changePixmap = pyqtSignal(QImage)

    def __init__(self, path = 0, parent=None):
        QThread.__init__(self, parent=parent)
        self.vid_src = path

    def setPath(self, path):
        self.vid_src = path

    def run(self):
        global carStartPosition
        global carCurrentPosition
        out, source, weights, view_vid, save_vid, imgsz = "output", self.vid_src, "yolov5/weights/yolov5s.pt", True, True, 640
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device('') # Default CUDA
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        print(device.type)
        # Set Dataloader
        vid_writer, vid_path = None, None
        # Check if environment supports image displays
        if view_vid:
            view_vid = check_imshow()

        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        save_path = str(Path(out))

        # Count frame
        count_frame = 0
        start_time = 0
        end_time = 0

        while(flag):
            for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                start_time = time.time()
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if not flag:
                    break
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                if(frame_idx >=5):
                    count_frame += 1

                # Inference
                pred = model(img, augment='')[0]

                # Apply NMS
                pred = non_max_suppression(pred, 0.4, 0.5, classes=[2, 5, 6], agnostic=False)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    else:
                        p, s, im0 = path, '', im0s

                    save_path = str(Path(out) / Path(videoFile))

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()

                        bbox_xywh = []
                        confs = []

                        # Adapt detections to deep sort input format
                        for *xyxy, conf, cls in det:
                            x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                            obj = [x_c, y_c, bbox_w, bbox_h]
                            bbox_xywh.append(obj)
                            if conf.item() > 0.55:
                                confs.append([conf.item()])

                        xywhs = torch.Tensor(bbox_xywh)
                        confss = torch.Tensor(confs)

                        # Pass detections to deepsort
                        outputs = deepsort.update(xywhs, confss, im0)

                        end_time = time.time()
                        # Calculate FPS
                        if not (start_time == end_time):
                            fps = 1.0/ (end_time - start_time)

                        # draw boxes for visualization
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]

                            # Calculate speed
                            if(count_frame == 1):
                                car_ID = outputs[:, 4]
                                post_car = outputs[:, [0, 1, 2, 3]]
                                for ID in car_ID:
                                    post = np.where(car_ID == ID)
                                    carStartPosition[ID] = post_car[int(post[0]), :].tolist()
                            if(count_frame == 2):
                                car_ID = outputs[:, 4]
                                post_car = outputs[:, [0, 1, 2, 3]]
                                for ID in car_ID:
                                    post = np.where(car_ID == ID)
                                    carCurrentPosition[ID] = post_car[int(post[0]), :].tolist()
                                for ID in carStartPosition.keys():
                                    if(ID in carStartPosition and ID in carCurrentPosition):
                                        [x_s1, y_s1, x_s2, y_s2] = carStartPosition[ID]
                                        [x_c1, y_c1, x_c2, y_c2] = carCurrentPosition[ID]
                                        #carStartPosition[ID] = [x_c1, y_c1, x_c2, y_c2]
                                        if ID not in speed:
                                            speed_car = calculate_speed([x_s1, y_s1, x_s2, y_s2], [x_c1, y_c1, x_c2, y_c2], fps=fps)
                                            speed[ID] = int(speed_car)
                                count_frame = 0

                            for id in violating_vehicle.keys():
                                img_crop = im0[violating_vehicle[id][1]:violating_vehicle[id][3], violating_vehicle[id][0]:violating_vehicle[id][2]]
                                if(img_crop.shape[0] > 0  and img_crop.shape[1]>0):
                                    if not os.path.isfile(path_save_violating_vehicle + "{}.{}.jpg".format(fileName, id)):
                                        cv2.imwrite(path_save_violating_vehicle + "{}.{}.jpg".format(fileName, id), img_crop)
                                        imgFile = fileName + "." + str(id) + "." + "jpg"
                                        sp = "{} km/h".format(speed[id])
                                        SQL.queryNoReturn("Insert Into ViolatingVehicle Values ('{}', '{}', '{}', '{}', '{}')".format(id, sp, violating_name[id], violation_time[id], imgFile))
                    
                            draw_boxes(im0, bbox_xyxy, identities, speed=speed)
                    else:
                        deepsort.increment_ages()

                    # Stream results
                    if view_vid:
                        im0S = cv2.resize(im0, (850, 480))
                        rgbImage = cv2.cvtColor(im0S, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgbImage.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(850, 480, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)

                    if save_vid:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                print(save_path)
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            print(save_path)
                        vid_writer.write(im0)

            
class Monitoring(QFrame, Ui_Monitoring):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        self.setupUi(self)
        self.comboInput.addItem("Video")
        self.comboInput.addItem("Camera")
        self.txtPPM.setText("1")
        self.btEnd.setEnabled(False)
        self.path = None

        # Monitoring Video Thread
        self.threadMonitoring = ThreadMonitoring(self)
        self.threadMonitoring.changePixmap.connect(self.setImage)

        # Event
        self.btChooseVideo.clicked.connect(self.chooseVideo)
        self.btStart.clicked.connect(self.startMonitoring)
        self.btEnd.clicked.connect(self.endMonitoring)
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.lbImg.setPixmap(QPixmap.fromImage(image))

    def alert(self, title, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def chooseVideo(self):
        #Show file dialog
        global fileName
        global videoFile
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            video_path, _ = QFileDialog.getOpenFileName(None, "Choose Video", "","Video Files (*.mp4 *.avi);;All Files (*)", options = options)
            if video_path is not None:
                self.path = video_path
                fileName = Path(video_path).resolve().stem
                fileTail = os.path.splitext(video_path)[-1]
                videoFile = fileName + fileTail
                self.lbNameVideo.setText(videoFile)
        except:
            pass
    
    def startMonitoring(self):
        global flag
        global pixels_per_meter
        global fileName
        global videoFile
        if not self.txtPPM.text() == "":
            if(float(self.txtPPM.text()) > 0):
                pixels_per_meter = float(self.txtPPM.text())
                #print(pixels_per_meter)
            else:
                self.alert(title="Cảnh báo", message="Pixel trên met phải lớn hơn 0, \n Hệ thống sẽ cài đặt mặc định")
        else:
            pixels_per_meter = 1
        if(self.comboInput.currentText() == "Camera"):
            if not flag:
                fileName = "webcam0"
                videoFile = "Webcam0.mp4"
                flag = True
                self.threadMonitoring.setPath(path="0")
                self.threadMonitoring.start()
        if(self.path is None and self.comboInput.currentText() == "Video"):
            self.alert(title="Cảnh báo", message="Không có vide được chọn!")
        else:
            if not flag:
                self.btStart.setEnabled(False)
                self.btEnd.setEnabled(True)
                flag = True
                self.threadMonitoring.setPath(path=self.path)
                self.threadMonitoring.start()
                
    def endMonitoring(self):
        global flag
        if flag:
            flag = False
            self.threadMonitoring.exit()
            self.btEnd.setEnabled(False)
            self.btStart.setEnabled(True)
            if not flag:
                cleanDictSpeed()