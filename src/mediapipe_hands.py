# src/mediapipe_hands.py
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=4,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

def get_hands(frame):
    """
    returns list of hand bounding boxes as [x1,y1,x2,y2] in pixel coords
    and list of landmarks per hand (normalized)
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    out_boxes = []
    out_landmarks = []
    if res.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            out_boxes.append([x1, y1, x2, y2])
            out_landmarks.append(hand_landmarks)
    return out_boxes, out_landmarks

def ball_overlaps_hand(ball_bbox, hand_bbox, iou_threshold=0.02):
    # simple IoU for ball-hand overlap, low threshold
    xA = max(ball_bbox[0], hand_bbox[0])
    yA = max(ball_bbox[1], hand_bbox[1])
    xB = min(ball_bbox[2], hand_bbox[2])
    yB = min(ball_bbox[3], hand_bbox[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return False
    boxAArea = (ball_bbox[2]-ball_bbox[0]) * (ball_bbox[3]-ball_bbox[1])
    boxBArea = (hand_bbox[2]-hand_bbox[0]) * (hand_bbox[3]-hand_bbox[1])
    union = boxAArea + boxBArea - interArea
    return (interArea / union) >= iou_threshold
