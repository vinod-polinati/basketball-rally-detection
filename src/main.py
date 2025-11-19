import cv2
import numpy as np
from detect import detect
from tracker import track_players, track_ball
from annotate import draw_box, draw_shot_banner

# ------------------------------
# Helper functions
# ------------------------------

def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


# ------------------------------
# Parameters for shooter detection
# ------------------------------

POSSESSION_IOU_THRESH = 0.15
POSSESSION_DIST_THRESH = 120.0
POSSESSION_FRAMES_REQUIRED = 2

SHOT_VELOCITY_FRAMES = 4
SHOT_UP_VELOCITY_THRESH = -6  # negative = ball going upward

ball_history = []
possession_history = []
frame_idx = 0
shooter_id = None


# ------------------------------
# Video I/O
# ------------------------------

video = cv2.VideoCapture("video.mp4")
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      30, (w, h))

print("Processing video...")


# ------------------------------
# Main Loop
# ------------------------------

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Step 1: YOLO detection
    players, balls = detect(frame)

    # Step 2: Norfair tracking
    tracked_players = track_players(players)   # [(track, bbox)]
    tracked_ball = track_ball(balls)           # [(track, bbox)]

    # ------------------------------
    # Step 3: Shooter detection logic
    # ------------------------------

    if tracked_ball:
        ball_track, ball_bbox = tracked_ball[0]
        bcx, bcy = ball_track.estimate[0]

        # Store ball history — safe version
        ball_history.append((frame_idx, float(bcx), float(bcy), ball_bbox))

        # Find closest player
        closest_id = None
        closest_score = 1e9
        closest_bbox = None

        for track, pbbox in tracked_players:
            pcx, pcy = track.estimate[0]
            dist = np.hypot(pcx - bcx, pcy - bcy)
            overlap = iou(pbbox, ball_bbox)
            score = dist - (overlap * 200)
            if score < closest_score:
                closest_score = score
                closest_id = track.id
                closest_bbox = pbbox

        # Possession check
        possessor = None
        if closest_id is not None:
            if iou(closest_bbox, ball_bbox) >= POSSESSION_IOU_THRESH or closest_score < POSSESSION_DIST_THRESH:
                possessor = closest_id

        possession_history.append((frame_idx, possessor))

        # ---- BALL UPWARD MOTION ----
        if len(ball_history) >= SHOT_VELOCITY_FRAMES:
            y_now = ball_history[-1][2]
            y_past = ball_history[-SHOT_VELOCITY_FRAMES][2]

            # If any missing data → skip safely
            if y_now is not None and y_past is not None:
                delta_y = y_now - y_past
                velocity = delta_y / SHOT_VELOCITY_FRAMES

                if velocity < SHOT_UP_VELOCITY_THRESH:
                    # Look back recent frames
                    lookback = POSSESSION_FRAMES_REQUIRED + 3
                    recent = [p for _, p in possession_history[-lookback:] if p is not None]

                    if recent:
                        candidate = max(set(recent), key=recent.count)
                        if recent.count(candidate) >= POSSESSION_FRAMES_REQUIRED:
                            shooter_id = candidate
                            possession_history = []  # avoid double-counting
                            print(f"[Frame {frame_idx}] SHOT ATTEMPT BY Player {shooter_id} | vel={velocity:.2f}")

    else:
        # No ball detected → repeat last known coords if exists
        if ball_history:
            last_frame, last_x, last_y, last_box = ball_history[-1]
            ball_history.append((frame_idx, last_x, last_y, last_box))
        else:
            ball_history.append((frame_idx, None, None, None))
        possession_history.append((frame_idx, None))

    # ------------------------------
    # Step 4: Draw players
    # ------------------------------

    for track, bbox in tracked_players:
        color = (0, 255, 0)
        label = f"Player {track.id}"

        if shooter_id == track.id:
            color = (0, 0, 255)
            label = f"Shooter {track.id}"

        draw_box(frame, bbox, color, label)

    # ------------------------------
    # Step 5: Draw ball
    # ------------------------------

    if tracked_ball:
        ball_track, ball_bbox = tracked_ball[0]
        draw_box(frame, ball_bbox, (255, 255, 0), "Ball")

    # OPTIONAL: show banner if shooter identified this frame
    if shooter_id is not None:
        draw_shot_banner(frame, "SHOT ATTEMPT")

    # ------------------------------
    # Step 6: Write frame
    # ------------------------------

    out.write(frame)
    frame_idx += 1


# ------------------------------
# Cleanup
# ------------------------------

video.release()
out.release()
print("DONE — output.mp4 saved!")
