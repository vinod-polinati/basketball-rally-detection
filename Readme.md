# Basketball Video Analytics Pipeline

**_Automated Player Tracking, Shooter Detection, Pass Recognition & Shot Outcome Analysis_**

## Overview

This project processes basketball match videos and automatically performs:

- Player detection
- Ball detection
- Multi-object tracking
- Shooter identification (hand gesture + ball trajectory)
- Pass detection
- Shot trajectory analysis
- Shot success prediction
- Annotated output video generation

The pipeline combines **YOLOv8**, **Norfair tracking**, **MediaPipe Hands**, and custom logic to deliver professional-level sports analytics.

---

# Installation & Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/vinod-polinati/basketball-rally-detection.git
cd basketball-rally-detection
```

---

## 2. Create a Virtual Environment (Python 3.11)

If you installed Python 3.11 via Homebrew, run:

```bash
python3.11 -m venv venv
```

Activate it:

### macOS / Linux:

```bash
source venv/bin/activate
```

### Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

Make sure your virtual environment is activated, then run:

```bash
pip install -r requirements.txt
```

---

## 5. Run the Pipeline

Now run the main script:

```bash
python3 src/main.py
```

This will:

- Process your video frame-by-frame
- Detect players and ball
- Track IDs
- Identify shooter & passes
- Analyze shot trajectory
- Draw annotations
- Save output

---

## 6. Output Video

After processing, the program generates:

```
output.mp4
```

This file contains all overlays:

- Player IDs
- Shooter highlights
- Pass lines
- Ball arc
- Shot result
- Bounding boxes

## Features

- **Player & Ball Detection** using YOLOv8
- **Consistent Tracking (IDs)** using Norfair
- **Shooter Identification** using MediaPipe hand landmarks
- **Trajectory Analysis** for shots
- **Pass Detection** via ball direction change
- **Fully Annotated Video Output**
- Modular code structure for easy improvements

---

## System Architecture

```
Input Video
     │
     ▼
YOLOv8 Detection  → players, ball
     │
     ▼
Norfair Tracking  → stable object IDs
     │
     ├──► MediaPipe Hands (determine shooter)
     │
     ├──► Ball Trajectory Analysis (shot detection)
     │
     └──► Pass Detection (direction change)
     ▼
Annotation Engine
     ▼
Output Video (with analytics overlays)
```

---

## Project Structure

```
statslane-asngmt/
│
├── myenv/                     # Virtual environment
│
├── src/
│   ├── annotate.py             # Drawing utilities (boxes, text, labels)
│   ├── detect.py               # YOLO detection logic
│   ├── main.py                 # Full end-to-end analytics pipeline
│   ├── mediapipe_hands.py      # Hand gesture detection (shooting pose)
│   └── tracker.py              # Norfair tracking logic
│
├── video.mp4                  # <-- place your input video here
├── output.mp4                 # Generated after running code
├── Readme.md
└── requirements.txt

```

---

## Core Components

### 1. YOLOv8 — Player & Ball Detection

- Detects class **0 (person)** → players
- Detects class **32 (sports ball)** → basketball
- Converts YOLO boxes into usable (x1, y1, x2, y2) format

### 2. Norfair Tracking

- Assigns persistent IDs to players and the ball
- Enables:

  - Shooter identification
  - Pass recognition
  - Possession analysis
  - Trajectory tracking

### 3. MediaPipe Hands

- Extracts player hand region
- Detects hand landmarks
- Recognizes **shot release** gesture
- Helps confirm shooter when ball is close

### 4. Ball Trajectory Analysis

- Tracks ball path
- Determines:

  - Shot start
  - Peak
  - Descent
  - Shot result (made/missed when entering hoop region)

### 5. Pass Detection

Triggered when ball direction changes sharply:

- Pre-change nearest player → **passer**
- Post-change nearest player → **receiver**

---

## Output

The system overlays:

- Player IDs
- Ball box
- Shooter highlight
- Pass arrows
- Ball trajectory arc
- Shot result ("Made" / "Missed")

Result is saved as an annotated output video.

---

## Future Improvements

- Train custom action recognition model (shoot/pass/dribble)
- Add player re-identification (jersey numbers or embeddings)
- Use pose estimation for better shooter tracking
- Add team possession + event timeline export
