# OpenCV Virtual Drawing & Shape Studio

**Variant 9 — Advanced OpenCV Laboratory**

A gesture-controlled virtual drawing workstation built with Python, OpenCV, and MediaPipe Hands. Extends a pre-lab media controller with a full drawing studio interface.

## Overview

The system allows users to draw, create shapes, and manipulate a virtual canvas using only hand gestures captured via webcam. The existing pre-lab gesture controls (volume, play/pause, track navigation, mute, lock/unlock) are preserved and operate alongside the new drawing tools.

## Features

### Pre-Lab (Preserved)
- Volume control via thumb-index finger distance
- Play/Pause via open palm
- Next/Previous track via swipe gestures
- Mute/Unmute via closed fist
- Lock/Unlock via two-finger gesture (with confirmation)

### New Drawing Features
- Freehand drawing with adjustable brush size
- Eraser tool
- 10-color palette
- Straight line tool (point-to-point)
- Rectangle tool (outline)
- Circle tool (center + radius)
- Undo (history-based)
- Clear canvas (with confirmation)
- Save to PNG file with timestamped filename
- Gesture-controlled toolbar with hover feedback

## Technologies

- **Python 3.x**
- **OpenCV** — camera capture, image rendering, canvas manipulation
- **MediaPipe Hands** — hand landmark detection (21 landmarks per hand)
- **NumPy** — array operations for canvas management

## Architecture

```
Camera Input
    ↓
MediaPipe Hand Detection
    ↓
Landmark Extraction
    ↓
Gesture Recognition (static + motion-based)
    ↓
State Manager (LOCKED / MEDIA / DRAWING)
    ↓
Tool / Color / Size Selection (gesture-controlled toolbar)
    ↓
Drawing Engine (freehand, shapes, eraser)
    ↓
Canvas (white background, separate from webcam feed)
    ↓
History Manager (undo stack)
    ↓
OpenCV Dashboard (mode, gesture, tool, FPS, actions)
    ↓
Camera Display (composite frame)
```

## Gesture Controls

| Gesture | Pre-Lab Function | Drawing Function |
|---------|-----------------|------------------|
| Thumb-Index Distance | Volume control | — |
| Open Palm | Play/Pause (media) | Save canvas (when in upper area) |
| Right Swipe | Next track | — |
| Left Swipe | Previous track | — |
| Closed Fist | Mute/Unmute (media) | Toggle Drawing/Media mode (dwell) |
| Two Fingers | Lock/Unlock (confirm) | — |
| Index Point (index up, others down) | — | Freehand draw / place shape |
| Pinch (thumb touches index) | — | Start/commit geometric shape |
| Three Fingers | — | Undo |
| Four Fingers | — | Clear canvas (hold to confirm) |
| Open Palm (top of screen) | — | Save |

### Landmarks Used

| Landmark | Description | Use Case |
|----------|-------------|----------|
| 0 | Wrist | Swipe tracking, position reference |
| 4 | Thumb tip | Pinch detection, distance calculation |
| 8 | Index fingertip | Pointing gesture, drawing cursor |
| 12 | Middle fingertip | Finger count, gesture recognition |
| 16 | Ring fingertip | Finger count, gesture recognition |
| 20 | Pinky fingertip | Finger count, gesture recognition |

## Drawing Tools

| Tool | Function | Interaction |
|------|----------|-------------|
| Freehand (DRAW) | Continuous stroke | Index point + move |
| Erase | Remove pixels | Index point + move over canvas |
| Line | Straight line | Pinch at start → move → pinch at end |
| Rectangle | Outline rectangle | Pinch at corner → move → pinch opposite corner |
| Circle | Circle by radius | Pinch at center → move → pinch to set radius |
| Undo | Restore previous state | Three fingers |
| Clear | Empty canvas | Four fingers (hold 0.5s to confirm) |
| Save | Export as PNG | Open palm in upper screen area |

## Installation

```bash
# Navigate to project directory
cd OIRA_SOUTHVILLE_HANDGESTURE

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

### Pre-Lab (Media Controller)
```bash
python3 prelab/handgesture_prelab.py
```

### Lab (Drawing Studio)
```bash
python3 lab/handgesture_lab.py
```

### Emergency Keyboard Controls
| Key | Function |
|-----|----------|
| `q` | Quit |
| `m` | Toggle between Media and Drawing mode |
| `Esc` | Quit |

## Testing Strategy

Tests cover:
1. **Pre-Lab regression** — volume, play/pause, next/prev, mute, lock/unlock
2. **Drawing tools** — freehand, eraser, colors, thickness, line, rect, circle, undo, clear, save
3. **Robustness** — invalid gestures, wrong mode, accidental toolbar touch, cooldown, confirmation, empty history, save failure, webcam failure

Each test should be documented with: Test ID, Feature, Initial State, Gesture/Input, Expected Result, Actual Result, Pass/Fail, Notes.

## Limitations

- **Lighting**: Poor lighting reduces hand detection accuracy; consistent frontal lighting recommended
- **Camera quality**: Lower resolution cameras produce noisier landmarks
- **Hand occlusion**: Objects blocking the hand may cause tracking loss
- **Gesture ambiguity**: Similar poses (e.g., three vs four fingers) require clear finger extension
- **Jitter**: Fast movements may cause slight cursor overshoot; smoothed by exponential moving average
- **Threshold sensitivity**: Swipe threshold (80px) and confirmation time (0.5s) may need tuning per user preference
- **Single hand**: Only one hand is tracked at a time

## Future Improvements

- Filled shapes (rectangle, circle)
- Text input tool
- Polygon drawing tool
- Color picker from canvas
- Redo functionality
- Canvas resize / zoom
- Gesture-controlled brush size (continuous via distance)
- Multi-hand support
- Background removal for transparent canvas
- Real-time collaboration

## File Structure

```
OIRA_SOUTHVILLE_HANDGESTURE/
├── prelab/
│   └── handgesture_prelab.py
├── lab/
│   └── handgesture_lab.py
├── surprise_modification/
├── output/
├── tests/
    └── test_gestures.py
    └── test_prelab.py
├── docs/
    └── lab 1.pdf
├── requirements.txt
├── .gitignore
└── README.md
└── run.sh
```


