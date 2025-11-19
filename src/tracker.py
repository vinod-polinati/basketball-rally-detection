# src/tracker.py
from norfair import Detection, Tracker
import numpy as np

def bbox_to_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array([[cx, cy]])

# tweak thresholds if needed
player_tracker = Tracker(distance_function="mean_euclidean", distance_threshold=50)
ball_tracker   = Tracker(distance_function="mean_euclidean", distance_threshold=80)

def track_players(players):
    """
    players: list of [x1,y1,x2,y2]
    returns list of (track, bbox)
    """
    detections = [Detection(bbox_to_center(p), data=p) for p in players]
    tracks = player_tracker.update(detections)
    return [(t, t.last_detection.data) for t in tracks]

def track_ball(balls):
    """
    balls: list of [x1,y1,x2,y2]
    returns list of (track, bbox)
    """
    detections = [Detection(bbox_to_center(b), data=b) for b in balls]
    tracks = ball_tracker.update(detections)
    return [(t, t.last_detection.data) for t in tracks]
