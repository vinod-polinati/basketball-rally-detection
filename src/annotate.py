# src/annotate.py
import cv2

def draw_box(frame, box, color, text=None, thickness=2):
    """
    box: [x1,y1,x2,y2]
    color: (B,G,R)
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if text:
        # draw filled background for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_shot_banner(frame, text="SHOT ATTEMPT", alpha=0.8):
    """
    Draw a semi-transparent banner at top
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = int(h * 0.08)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 255), -1)  # red banner
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # text centered
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    x = int((w - tw) / 2)
    y = int(bar_h / 2 + th / 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
