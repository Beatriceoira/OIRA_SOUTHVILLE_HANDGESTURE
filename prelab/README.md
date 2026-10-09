## Features

### Pre-Lab (Preserved)
- Volume control via thumb-index finger distance
- Play/Pause via open palm
- Next/Previous track via swipe gestures
- Mute/Unmute via closed fist
- Lock/Unlock via two-finger gesture (with confirmation)

How the Pre-Lab (`prelab/handgesture_prelab.py`) implements them:

| Gesture | Pre-Lab behavior |
|---------|------------------|
| Thumb-index distance | Volume 0 to 100 (distance 0.02 to 0.3). Applied only when the value changes by more than 3 and the 0.7 s cooldown is over. Lowest priority: used only when no other gesture matches |
| Open palm | Play/Pause after a 0.5 s confirmation (progress bar). The pose must be released and formed again to repeat |
| Swipe right / left | Next / Previous track. Time-based: at least 60 px sideways, mostly horizontal, within 0.6 s, at 80 px/s or faster |
| Closed fist | Mute / Unmute immediately (no hold). Holding the fist toggles again every 0.7 s |
| Two fingers | Lock / Unlock after a 0.5 s confirmation. Works while locked; every other gesture is ignored while locked |

Priority when several gestures could match: two fingers, swipe, open palm, fist, volume. The Pre-Lab has no modes, only the lock flag, and its media control is Linux only (`playerctl`, `pactl`, `wpctl`).

### Lab: Two Modes
The lab starts **LOCKED**. Hold two fingers to unlock; it returns to the last mode used (MEDIA the first time). A closed fist held for 0.5 s switches between the two modes, and each mode only shows its own controls, detections, log lines and overlays.

**MEDIA mode (audio / playback)**
- Volume, play/pause, next/previous track and mute by gesture, carried over from the Pre-Lab with refined timing (palm held still 0.4 s, 5-frame flick, fist tap for mute; see the table in the Gesture Controls section)
- On-screen media buttons (PREV, PLAY/PAUSE, NEXT, MUTE, VOL-, VOL+), clickable with the mouse
- Volume bar and a live thumb-index line while adjusting volume

**DRAWING mode**
- Freehand drawing with adjustable brush size (2-50 px), eraser tool
- 10-color palette
- Straight line, rectangle (outline) and circle (center + radius) tools
- Undo **and Redo** (history-based, 30 steps)
- Clear canvas (hold to confirm), save to PNG with a timestamped filename
- Slim top bar that always shows the current tool / color / size and holds UNDO, REDO, CLEAR, SAVE
- **Collapsible bars** for tools/shapes, colors and brush sizes; all start closed to keep the screen uncluttered and open or close independently
- Gesture shortcuts for everything: flick to change color or tool, horns to set brush size, three fingers to undo/redo
- Buttons can be pressed by hovering the index finger, by mouse click, or by keyboard

### Lab: Everywhere
- **Window resize gesture** (distance-based): spread thumb and pinky (shaka sign) to enlarge the window, bring them together to shrink it. The size is capped to your screen where it can be detected. `+` / `-` and `f` (fullscreen) also work.
- **Hand controls can be switched off and on** one by one from the CONTROLS panel (or `g`), for example to turn off drawing while you only want to control music. Hand controls pause while that panel is open.
- **Status panel** at the bottom (mode, gesture, tool, FPS, action log) is **resizable and foldable**: drag its top edge, click its arrow, or use `v` and `<` `>`.
- **Cross-platform**: Windows, macOS and Linux (see [Platform Support](#platform-support)).
- Mode-aware gesture recognition: a pose that does nothing in the current mode is reported as Idle, and disabled controls are marked "(OFF)".
- Help overlay (`h`) listing the controls for the current mode.

## Technologies

- **Python 3.x**
- **OpenCV**: camera capture, image rendering, canvas manipulation, window and mouse handling
- **MediaPipe Hands**: hand landmark detection (21 landmarks per hand)
- **NumPy**: array operations for canvas management

## Architecture

```
Camera Input (platform-specific backend: V4L2 / DirectShow / AVFoundation)
    ↓
MediaPipe Hand Detection
    ↓
Landmark Extraction
    ↓
Gesture Recognition (static poses + flick/motion + distance measures)
    ↓
Mode filter + Hand-control switches (only enabled, mode-relevant gestures act)
    ↓
State Manager (LOCKED / MEDIA / DRAWING)
    ├── MEDIA:   Media Controller (Windows / macOS / Linux backends)
    └── DRAWING: Toolbar (collapsible bars) → Drawing Engine (freehand, shapes, eraser)
                      ↓
                 Canvas (white background, separate from webcam feed)
                      ↓
                 History Manager (undo + redo stacks)
    ↓
Window Manager (gesture / keyboard resize, fullscreen)
    ↓
OpenCV Dashboard (foldable, resizable status panel) + Controls panel + Help overlay
    ↓
Camera Display (composite frame)
```

## Gesture Controls

Only one hand is tracked. Poses are classified from which of the index, middle, ring and pinky fingers are extended, plus a thumb check where needed.

| Gesture | MEDIA mode | DRAWING mode |
|---------|-----------|--------------|
| Index point + thumb-index distance | Volume | n/a |
| Index point | n/a | Draw / erase / position a shape; aim at a top-bar button (hold 0.4 s to press) |
| Pinch (thumb touches index) | n/a | Commit the line / rectangle / circle |
| Open palm (hold still) | Play/Pause | Save canvas (when in the upper third of the screen) |
| Open hand flick right / left | Next / Previous track | Next / Previous **color** |
| Open hand flick up / down | n/a | Previous / Next **tool** (draw, erase, line, rect, circle) |
| Three fingers | n/a | Hold or flick left = Undo, flick right = **Redo** |
| Four fingers (hold still) | n/a | Clear canvas (hold to confirm) |
| Horns (index + pinky up) | n/a | **Brush size** by the gap between the two fingers |
| Quick fist tap | Mute / Unmute | n/a |
| Closed fist (hold 0.5 s) | Switch to DRAWING | Switch to MEDIA |
| Two fingers (hold 0.5 s) | Lock / Unlock | Lock / Unlock |
| Shaka (thumb + pinky out) | **Window size** by the thumb-pinky span | **Window size** by the thumb-pinky span |

Hold-type actions (clear, save, play/pause, undo hold) only count while the hand is still, so a flick cannot trigger them by accident. After pressing a top-bar button by finger, drawing is paused for 0.5 s so a stray mark is not left when the bars change size.

### Pre-Lab Gestures in the Lab

The Lab keeps all five Pre-Lab gestures but changes how some work:

| Gesture | Pre-Lab | Lab |
|---------|---------|-----|
| Volume | Change of more than 3, outside the cooldown; value cut to a whole number | Change of at least 3, at most one command every 0.12 s; value rounded; only while pointing with the index finger |
| Play / pause | 0.5 s confirmation after the open palm appears | Open palm held still for 0.4 s |
| Next / previous | Time-based: 60 px within 0.6 s at 80 px/s | Wrist moves 80 px within the last 5 frames |
| Mute | Fist fires at once | Fist closed and released within 0.1 to 0.5 s; holding 0.5 s switches mode |
| Lock / unlock | Two fingers held 0.5 s | Same, and unlocking returns to the previous mode |

### Landmarks Used

| Landmark | Description | Use Case |
|----------|-------------|----------|
| 0 | Wrist | Flick tracking, palm size (with 9) |
| 3, 4 | Thumb IP / tip | Thumb-extended check, pinch, thumb-index distance, shaka span |
| 8 | Index fingertip | Pointing gesture, drawing cursor, horns span |
| 12 | Middle fingertip | Finger count, gesture recognition |
| 16 | Ring fingertip | Finger count, gesture recognition |
| 17 | Pinky base | Thumb-extended check |
| 20 | Pinky tip | Finger count, horns span, shaka span |

The horns and shaka distances are divided by the palm length (wrist to middle-finger base), so they do not depend on how far you sit from the camera.

## Drawing Tools

| Tool | Function | Interaction |
|------|----------|-------------|
| Freehand (DRAW) | Continuous stroke | Index point + move |
| Erase | Remove pixels | Index point + move over canvas |
| Line | Straight line | Index point to set the start, move to the end, pinch to commit |
| Rectangle | Outline rectangle | Index point to set a corner, move to the opposite corner, pinch to commit |
| Circle | Circle by radius | Index point to set the center, move to set the radius, pinch to commit |
| Undo | Restore previous state | Three fingers (hold or flick left), UNDO button, `u` |
| Redo | Re-apply an undone step | Three fingers flick right, REDO button, `r` |
| Clear | Empty canvas | Four fingers (hold 0.5 s to confirm), CLEAR button, `c` |
| Save | Export as PNG | Open palm in the upper screen area, SAVE button, `s` |

Brush sizes can be picked from the SIZE bar (2, 5, 10, 20, 30, 50 px) or set continuously with the horns gesture. The eraser has a fixed size.

## Controls Beyond Gestures

### Mouse
| Action | Function |
|--------|----------|
| Click a top-bar button | TOOL / COLOR / SIZE open or close their bar; UNDO, REDO, CLEAR, SAVE act immediately |
| Click a tool, color or size | Select it |
| Click a media button (MEDIA mode) | PREV, PLAY/PAUSE, NEXT, MUTE, VOL-, VOL+ |
| Drag the status panel's top edge | Resize the panel (drag it to the bottom to fold it) |
| Click the status panel arrow | Fold / unfold the panel |
| Click CONTROLS | Open the hand-controls panel and tick controls on or off |

### Keyboard
| Key | Function |
|-----|----------|
| `q` / `Esc` | Quit |
| `m` | Toggle between Media and Drawing mode |
| `l` | Lock / unlock |
| `f` | Fullscreen |
| `+` / `-` | Window larger / smaller |
| `h` | Show / hide the help overlay |
| `v` | Fold / unfold the status panel |
| `<` / `>` | Make the status panel shorter / taller |
| `g` | Open / close the hand-controls panel (`a` = switch all on) |
| `p` `n` `b` `x` | MEDIA: play/pause, next, previous, mute |
| `[` `]` | MEDIA: volume down / up. DRAWING: previous / next brush size |
| `u` `r` `c` `s` | DRAWING: undo, redo, clear, save |
| `d` `e` `t` | DRAWING: draw tool, erase tool, next tool |
| `,` `.` | DRAWING: previous / next color |
| `1` `2` `3` | DRAWING: open / close the tool, color, size bar |

The lock, mode and quit keys always work, even if the matching hand control is switched off.

## Hand-Control Switches

Open **CONTROLS** (status panel strip, or `g`) to tick controls on or off. The list shows the controls of the current mode plus the ones that work everywhere:

- Everywhere: fist-hold mode switch, lock/unlock, shaka window size
- MEDIA: volume, play/pause, track flick, mute
- DRAWING: index drawing, pressing top bars by finger, pinch, undo/redo, color flick, tool flick, horns brush size, clear, save

A disabled control does nothing. Unlocking by gesture always works even if the lock control is off. Settings are not saved between runs.

## Platform Support

| | Windows | macOS | Linux |
|---|---------|-------|-------|
| Camera backend | DirectShow, then Media Foundation | AVFoundation | V4L2 (`/dev/video N`) |
| Play / pause, next, previous | Media keys | System media keys (needs `pyobjc`), else `nowplaying-cli`, else the Spotify or Music app | `playerctl` (tries `MEDIA_PLAYER` first, then any player), else `xdotool` |
| Volume / mute | `pycaw` for exact volume, otherwise volume keys | `osascript` | `pactl`, else `wpctl`, else `amixer` |

If a backend is missing, the matching action reports "failed" in the log and the rest keeps working. On macOS, grant camera access to your terminal or IDE, and Accessibility access if you use media keys.

## Installation

```bash
# Navigate to project directory
cd OIRA_SOUTHVILLE_HANDGESTURE

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` needs at least `opencv-python`, `mediapipe` and `numpy`. Optional extras:

| Platform | Package / tool | Gives you |
|----------|----------------|-----------|
| Windows | `pycaw`, `comtypes` | Exact volume (otherwise volume keys are pressed) |
| macOS | `pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa` | System media keys |
| Any | `screeninfo` | Screen size for capping the window-resize gesture (automatic on Windows) |
| Linux | `playerctl`, plus `pactl` or `wpctl` or `amixer` | Media and volume control |

The script uses MediaPipe's classic `mp.solutions` API. If you get an error about `solutions`, install an older MediaPipe release (for example 0.10.21).

## Running

### Pre-Lab (Media Controller)
```bash
python3 prelab/handgesture_prelab.py
```

### Lab (Drawing Studio)
```bash
python3 lab/handgesture_lab.py
python3 lab/handgesture_lab.py --camera 1          # use another webcam
python3 lab/handgesture_lab.py --player spotify    # Linux: preferred playerctl player ('' = any)
```
On Windows use `python` instead of `python3`. The system starts locked: hold two fingers (or press `l`) to unlock. Saved drawings go to an `output/` folder next to the script.

## Configuration

These constants at the top of `handgesture_lab.py` are the ones you are most likely to tune:

| Constant | Default | Effect |
|----------|---------|--------|
| `CONFIRMATION_TIME` | 0.5 s | Hold time for lock, mode switch, clear, save |
| `HOVER_CONFIRM_TIME` | 0.4 s | Finger hover time to press a toolbar button |
| `COOLDOWN_TIME` | 0.7 s | Pause between repeated actions |
| `SWIPE_THRESHOLD` / `SWIPE_THRESHOLD_V` | 80 / 60 px | Size of a horizontal / vertical flick |
| `STILL_THRESHOLD` | 35 px | How still the hand must be for hold-type actions |
| `RESIZE_RATIO_MIN` / `MAX` | 0.9 / 2.0 | Thumb-pinky span (in palm lengths) for the smallest / largest window |
| `BRUSH_RATIO_MIN` / `MAX` | 0.5 / 1.4 | Index-pinky span (in palm lengths) for the smallest / largest brush |
| `WINDOW_SCALE_MIN` / `MAX` | 0.5 / 3.0 | Window size limits relative to the camera frame |
| `DASH_DEFAULT_H`, `DASH_MIN_H`, `DASH_MAX_H` | 200 / 96 / 320 px | Status panel height limits |
| `MEDIA_PLAYER` | `chromium` | Linux: player `playerctl` tries first |

## Testing Strategy

Tests cover:
1. **Pre-Lab regression**: volume, play/pause, next/prev, mute, lock/unlock (run on both the Pre-Lab and the Lab, whose timing differs)
2. **Drawing tools**: freehand, eraser, colors, thickness, line, rect, circle, undo, redo, clear, save
3. **Gesture shortcuts**: color flick, tool flick, horns brush size, three-finger undo/redo, shaka window resize
4. **UI**: collapsible bars, status panel resize and fold, CONTROLS panel, help overlay, mouse buttons, keyboard shortcuts
5. **Mode separation**: media items never shown in drawing mode and the reverse; Idle shown for gestures with no action in the current mode
6. **Hand-control switches**: each control off then on again; lock still unlocks when switched off; gestures paused while the CONTROLS panel is open
7. **Platform checks**: camera start, media and volume backends on Windows, macOS and Linux
8. **Robustness**: invalid gestures, wrong mode, accidental toolbar touch, cooldown, confirmation, empty undo/redo history, save failure, webcam failure, missing media backend

Each test should be documented with: Test ID, Feature, Initial State, Gesture/Input, Expected Result, Actual Result, Pass/Fail, Notes.

## Limitations

- **Lighting**: Poor lighting reduces hand detection accuracy; consistent frontal lighting recommended
- **Camera quality**: Lower resolution cameras produce noisier landmarks
- **Hand occlusion**: Objects blocking the hand may cause tracking loss
- **Gesture ambiguity**: Similar poses (e.g., three vs four fingers, open palm vs four fingers, which depends on the thumb) require clear finger extension; shaka and horns need the other fingers clearly curled
- **Jitter**: Fast movements may cause slight cursor overshoot; smoothed by exponential moving average
- **Threshold sensitivity**: Flick thresholds, hold times and the horns/shaka span ranges may need tuning per user and camera distance
- **Single hand**: Only one hand is tracked at a time
- **Window resize**: Depends on the OpenCV window backend; some Linux desktops (for example Wayland) may ignore programmatic resizing
- **Media control**: Needs the platform tools listed above; on macOS media keys need Accessibility permission, and Windows without `pycaw` can only step the volume
- **Settings**: Hand-control switches and panel sizes reset when the program restarts

## Future Improvements

- Filled shapes (rectangle, circle)
- Text input tool
- Polygon drawing tool
- Color picker from canvas
- Canvas resize / zoom
- Save hand-control switches and panel layout between runs
- Two-hand pinch-zoom for window size
- Multi-hand support
- Background removal for transparent canvas
- Real-time collaboration

Done since the first version: Redo, gesture-controlled brush size, window resize, cross-platform support.

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
│   └── test_gestures.py
│   └── test_prelab.py
├── docs/
│   └── HandGesture Studio - Code Reference.docx
│   └── lab 1.pdf
├── requirements.txt
├── .gitignore
└── README.md
└── run.sh
```


