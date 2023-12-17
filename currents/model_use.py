from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import ultralytics


def main(cur_model):
    def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(lw - 1, 1)  # font thickness

            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height

            outside = p1[1] - h >= 3

            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled

            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
        # Define COCO Labels
        if labels == []:
            labels = {0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane',
                      6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant',
                      12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog',
                      18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra',
                      24: u'giraffe',
                      25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee',
                      31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat',
                      36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle',
                      41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl',
                      47: u'banana',
                      48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
                      54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant',
                      60: u'bed',
                      61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote',
                      67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink',
                      73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear',
                      79: u'hair drier', 80: u'toothbrush'}
        # Define colors
        if colors == []:
            colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161),
                      (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123),
                      (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209),
                      (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126),
                      (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149),
                      (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233),
                      (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168),
                      (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0),
                      (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196),
                      (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237),
                      (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121),
                      (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5),
                      (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41),
                      (177, 184, 2), (35, 97, 56), (241, 22, 161)]

        for box in boxes:
            if score:
                if int(box[-1]) + 1 in labels:
                    label = labels[int(box[-1]) + 1] + " " + str(round(100 * float(box[-2]), 1)) + "%"
            else:
                if int(box[-1]) + 1 in labels:
                    label = labels[int(box[-1]) + 1]

            if conf:
                if box[-2] > conf:
                    color = colors[int(box[-1])]
                    if int(box[-1]) + 1 in labels:
                        box_label(image, box, label, color)
            else:
                color = colors[int(box[-1])]
                if int(box[-1]) + 1 in labels:
                    box_label(image, box, label, color)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    
    current, path_model = cur_model

    model = ultralytics.YOLO(path_model)

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(current)
    time.sleep(2.0)
    fps = FPS().start()


    
    
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
    
        (h, w) = frame.shape[:2]
        img = np.asarray(frame)
    
        pred_result = model.predict(img)
        img = plot_bboxes(img, pred_result[0].boxes.data, conf=0.3)
    
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break
    
        fps.update()
    
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    cv2.destroyAllWindows()