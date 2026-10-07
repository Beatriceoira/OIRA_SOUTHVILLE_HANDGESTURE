"""
handgesture_lab.py
OpenCV Virtual Drawing & Shape Studio - Gesture-Controlled
Runs on Windows, macOS and Linux.

Install (all OSes):   pip install opencv-python mediapipe numpy
Optional, per OS:
    Windows : pip install pycaw comtypes   (exact volume; otherwise volume keys are pressed)
    macOS   : pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa   (system media keys)
    Linux   : playerctl, plus pactl OR wpctl OR amixer (from your package manager)

TWO MODES  (Fist hold toggles between them)
  MEDIA mode
    Index point + thumb-index distance -> Volume
    Open palm (hold still)             -> Play / Pause
    Open hand flick right / left       -> Next / Previous track
    Quick fist tap                     -> Mute / Unmute
    Mouse: PREV | PLAY/PAUSE | NEXT | MUTE | VOL- | VOL+ buttons
  DRAWING mode
    Index point        -> Draw / erase / position shape; aim at a toolbar button to press it
    Pinch              -> Commit line / rect / circle
    Three fingers      -> hold or flick LEFT = Undo, flick RIGHT = Redo
    Open hand flick    -> left/right = previous/next COLOR, up/down = previous/next TOOL
    Horns (index+pinky)-> spread the two fingers = BRUSH SIZE
    Four fingers hold  -> Clear canvas
    Open palm (top)    -> Save canvas
    Mouse / finger-hover: top bar = TOOL, COLOR, SIZE (click to open or close that bar), UNDO REDO CLEAR SAVE
ANY mode
    Shaka (thumb+pinky out) -> spread thumb and pinky = WINDOW SIZE (distance based)
    Two fingers (hold)      -> Lock / Unlock

Status panel (bottom): drag its top edge to resize, click its arrow to fold / unfold.
CONTROLS button (or G): switch individual hand controls off / on (gestures pause while it is open).

Keyboard: q/Esc quit | m mode | l lock | f fullscreen | + / - window size | h help
    v fold status panel | < > resize it | g hand-controls panel (a = all on)
    MEDIA   : p play/pause, n next, b previous, x mute, [ ] volume
    DRAWING : u undo, r redo, c clear, s save, d draw, e erase, t next tool, , . color, [ ] size,
              1 / 2 / 3 open-close the tool / color / size bar
"""

import argparse
import ctypes
import math
import os
import platform
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================
CAMERA_INDEX = 0
CONFIRMATION_TIME = 0.5
COOLDOWN_TIME = 0.7
PALM_DWELL = 0.4
UNDO_HOLD = 0.3
SWIPE_THRESHOLD = 80        # px, horizontal flick over the last 5 frames
SWIPE_THRESHOLD_V = 60      # px, vertical flick
STILL_THRESHOLD = 35        # px, max wrist travel to count as "holding still"
SMOOTHING_FACTOR = 0.65
MIN_DISTANCE = 0.02
MAX_DISTANCE = 0.3
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 50
ERASER_SIZE = 40
MAX_HISTORY = 30
CANVAS_BACKGROUND = (255, 255, 255)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
HOVER_CONFIRM_TIME = 0.4
VOLUME_MIN_INTERVAL = 0.12  # s, keeps slow backends (osascript) from lagging the video
MEDIA_PLAYER = "chromium"   # Linux only: tried first with playerctl -p, then any player
DASH_DEFAULT_H = 200        # status panel height when open (drag its top edge to change it)
DASH_MIN_H = 96
DASH_MAX_H = 320
DASH_STRIP_H = 26           # always-visible strip; also the height of the folded panel
DRAG_GRAB = 6               # px around the panel's top edge that grab the resize handle
CONTROLS_BTN_W = 150

WINDOW_NAME = "HandGesture Studio - Lab"
WINDOW_SCALE_MIN = 0.5
WINDOW_SCALE_MAX = 3.0      # also capped by your screen size when it can be detected
# Spans are divided by palm length so they do not depend on how far you are from the camera.
RESIZE_RATIO_MIN = 0.9      # thumb-tip <-> pinky-tip span (shaka) that gives the smallest window
RESIZE_RATIO_MAX = 2.0      # ... and the largest window
BRUSH_RATIO_MIN = 0.5       # index-tip <-> pinky-tip span (horns) that gives MIN_BRUSH_SIZE
BRUSH_RATIO_MAX = 1.4       # ... and MAX_BRUSH_SIZE

COLORS = {
    "RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255), "CYAN": (255, 255, 0), "MAGENTA": (255, 0, 255),
    "WHITE": (255, 255, 255), "BLACK": (0, 0, 0), "ORANGE": (0, 165, 255),
    "PURPLE": (128, 0, 255),
}
COLOR_ORDER = list(COLORS.keys())
SHAPE_TOOLS = ["DRAW", "ERASE", "LINE", "RECT", "CIRCLE"]
TOOL_BUTTONS = SHAPE_TOOLS + ["UNDO", "REDO", "CLEAR", "SAVE"]
SIZES = [2, 5, 10, 20, 30, 50]
MEDIA_BUTTONS = ["PREV", "PLAY/PAUSE", "NEXT", "MUTE", "VOL-", "VOL+"]

MODE_MEDIA = "MEDIA"
MODE_DRAWING = "DRAWING"
MODE_LOCKED = "LOCKED"

SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MAC = SYSTEM == "Darwin"
IS_LINUX = not (IS_WINDOWS or IS_MAC)

FONT = cv2.FONT_HERSHEY_SIMPLEX

try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    raise SystemExit("This script needs MediaPipe's classic `solutions` API. "
                     "Install an older release, e.g.:  pip install mediapipe==0.10.21")


# ============================================================================
# SMALL HELPERS
# ============================================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def text_color_for(bgr):
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def put_centered(img, text, x1, x2, yc, scale, color, thickness=1):
    """Centre text in [x1, x2], shrinking it if it would not fit."""
    max_w = (x2 - x1) - 6
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    if tw > max_w:
        scale *= max_w / tw
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, (x1 + ((x2 - x1) - tw) // 2, yc + th // 2),
                FONT, scale, color, thickness, cv2.LINE_AA)


def put_outlined(img, text, org, scale, color, thickness=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_scale(lines, max_w, base=0.5):
    longest = max((cv2.getTextSize(l, FONT, base, 1)[0][0] for l in lines), default=1)
    return base * min(1.0, max_w / max(1, longest))


def get_screen_size():
    """Best-effort screen size without importing a GUI toolkit (None if unknown)."""
    if IS_WINDOWS:
        try:
            u = ctypes.windll.user32
            return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        except Exception:
            pass
    try:
        from screeninfo import get_monitors   # optional: pip install screeninfo
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return None


class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        self.value = v if self.value is None else self.alpha * v + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ============================================================================
# MEDIA CONTROLLER  (Windows / macOS / Linux; every backend is optional)
# ============================================================================
class MediaController:
    _WIN_VK = {"play_pause": 0xB3, "next": 0xB0, "prev": 0xB1,
               "mute": 0xAD, "vol_down": 0xAE, "vol_up": 0xAF}
    _MAC_KEY = {"play_pause": 16, "next": 17, "prev": 18}   # NX_KEYTYPE_*

    def __init__(self, player=MEDIA_PLAYER):
        self.player = player
        self.win_vol = None
        self._vol_est = 50
        self.have = {n: shutil.which(n) is not None for n in
                     ("playerctl", "pactl", "wpctl", "amixer", "xdotool",
                      "osascript", "nowplaying-cli")}
        if IS_WINDOWS:
            self._init_pycaw()
        found = [n for n, ok in self.have.items() if ok]
        if IS_WINDOWS:
            found = ["media keys"] + (["pycaw"] if self.win_vol is not None else [])
        print(f"Media backends ({SYSTEM}): {', '.join(found) if found else 'none found'}")

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _sh(cmd):
        """Run a command; stdout (str) on success, None on failure."""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    def _run(self, cmd):
        return self._sh(cmd) is not None

    # ---- Windows -----------------------------------------------------------
    def _init_pycaw(self):
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            dev = AudioUtilities.GetSpeakers()
            if hasattr(dev, "EndpointVolume"):          # newer pycaw
                self.win_vol = dev.EndpointVolume
            else:                                        # older pycaw
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.win_vol = ctypes.cast(iface, ctypes.POINTER(IAudioEndpointVolume))
        except Exception:
            self.win_vol = None

    def _win_key(self, name):
        try:
            user32 = ctypes.windll.user32
            vk = self._WIN_VK[name]
            user32.keybd_event(vk, 0, 0, 0)      # key down
            user32.keybd_event(vk, 0, 2, 0)      # key up
            return True
        except Exception:
            return False

    def _win_set_volume(self, pct):
        if self.win_vol is not None:
            try:
                self.win_vol.SetMasterVolumeLevelScalar(pct / 100.0, None)
                return True
            except Exception:
                pass
        steps = int(round((pct - self._vol_est) / 2.0))   # one volume key = 2 %
        name = "vol_up" if steps > 0 else "vol_down"
        for _ in range(min(abs(steps), 50)):
            if not self._win_key(name):
                return False
        return True

    # ---- macOS -------------------------------------------------------------
    @staticmethod
    def _mac_media_key(key):
        """Post a system media key (same as the keyboard's). Needs pyobjc."""
        try:
            from AppKit import NSEvent
            import Quartz
        except ImportError:
            return False
        try:
            for down in (True, False):
                flags = 0xA00 if down else 0xB00
                data1 = (key << 16) | ((0xA if down else 0xB) << 8)
                ev = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                    14, (0, 0), flags, 0, 0, 0, 8, data1, -1)
                Quartz.CGEventPost(0, ev.CGEvent())
            return True
        except Exception:
            return False

    def _mac_transport(self, name):
        if self._mac_media_key(self._MAC_KEY[name]):
            return True
        if self.have["nowplaying-cli"]:
            sub = {"play_pause": "togglePlayPause", "next": "next", "prev": "previous"}[name]
            if self._run(["nowplaying-cli", sub]):
                return True
        if self.have["osascript"]:
            verb = {"play_pause": "playpause", "next": "next track", "prev": "previous track"}[name]
            for app in ("Spotify", "Music"):
                out = self._sh(["osascript",
                                "-e", f'if application "{app}" is running then',
                                "-e", f'tell application "{app}" to {verb}',
                                "-e", 'return "ok"',
                                "-e", "end if",
                                "-e", 'return "no"'])
                if out == "ok":
                    return True
        return False

    # ---- Linux -------------------------------------------------------------
    def _linux_transport(self, name):
        action = {"play_pause": "play-pause", "next": "next", "prev": "previous"}[name]
        if self.have["playerctl"]:
            if self.player and self._run(["playerctl", "-p", self.player, action]):
                return True
            if self._run(["playerctl", action]):
                return True
        if self.have["xdotool"]:
            key = {"play_pause": "XF86AudioPlay", "next": "XF86AudioNext", "prev": "XF86AudioPrev"}[name]
            return self._run(["xdotool", "key", key])
        return False

    def _linux_set_volume(self, pct):
        if self.have["pactl"] and self._run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"]):
            return True
        if self.have["wpctl"] and self._run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct}%"]):
            return True
        if self.have["amixer"] and self._run(["amixer", "-q", "sset", "Master", f"{pct}%"]):
            return True
        return False

    # ---- public API --------------------------------------------------------
    def _transport(self, name):
        if IS_WINDOWS:
            return self._win_key(name)
        if IS_MAC:
            return self._mac_transport(name)
        return self._linux_transport(name)

    def play_pause(self):
        return self._transport("play_pause")

    def next_track(self):
        return self._transport("next")

    def previous_track(self):
        return self._transport("prev")

    def set_volume(self, pct):
        pct = int(clamp(pct, 0, 100))
        if IS_WINDOWS:
            ok = self._win_set_volume(pct)
        elif IS_MAC:
            ok = self.have["osascript"] and self._run(["osascript", "-e", f"set volume output volume {pct}"])
        else:
            ok = self._linux_set_volume(pct)
        if ok:
            self._vol_est = pct
        return bool(ok)

    def mute(self):
        if IS_WINDOWS:
            if self.win_vol is not None:
                try:
                    self.win_vol.SetMute(0 if self.win_vol.GetMute() else 1, None)
                    return True
                except Exception:
                    pass
            return self._win_key("mute")
        if IS_MAC:
            return bool(self.have["osascript"] and self._run(
                ["osascript", "-e", "set volume output muted (not (output muted of (get volume settings)))"]))
        if self.have["pactl"] and self._run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]):
            return True
        if self.have["wpctl"] and self._run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"]):
            return True
        return bool(self.have["amixer"] and self._run(["amixer", "-q", "sset", "Master", "toggle"]))

    def get_volume(self):
        """Best-effort current system volume 0-100, or None."""
        try:
            if IS_WINDOWS:
                if self.win_vol is not None:
                    return int(round(self.win_vol.GetMasterVolumeLevelScalar() * 100))
            elif IS_MAC:
                if self.have["osascript"]:
                    out = self._sh(["osascript", "-e", "output volume of (get volume settings)"])
                    return int(out) if out and out.isdigit() else None
            else:
                if self.have["pactl"]:
                    m = re.search(r"(\d+)%", self._sh(["pactl", "get-sink-volume", "@DEFAULT_SINK@"]) or "")
                    if m:
                        return int(m.group(1))
                if self.have["wpctl"]:
                    m = re.search(r"(\d+(?:\.\d+)?)", self._sh(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"]) or "")
                    if m:
                        return int(round(float(m.group(1)) * 100))
        except Exception:
            pass
        return None


media = MediaController()


# ============================================================================
# STATE / HISTORY / SHAPES
# ============================================================================
class StateManager:
    MODES = {MODE_LOCKED, MODE_MEDIA, MODE_DRAWING}

    def __init__(self):
        self.mode = MODE_LOCKED
        self.resume_mode = MODE_MEDIA      # mode restored on unlock
        self.system_locked = True
        self.volume = 50
        self.action_history = deque(maxlen=30)
        self.last_action_time = 0.0
        self.last_volume_time = 0.0
        self.swipe_history = deque(maxlen=5)   # (wrist_x, wrist_y) px
        self.toast = ""
        self.toast_time = 0.0
        self.toast_kind = "sys"
        self.draw_block_until = 0.0
        self.show_help = True

    def in_cooldown(self):
        return time.time() - self.last_action_time < COOLDOWN_TIME

    def log(self, action, kind="sys"):
        """Record an action WITHOUT starting the cooldown. kind: 'sys' | 'media' | 'draw'."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.action_history.append((ts, action, kind))
        print(f"[{ts}] {action}")

    def trigger_action(self, action, kind="sys"):
        self.last_action_time = time.time()
        self.log(action, kind)

    def notify(self, msg, kind="sys"):
        self.toast, self.toast_time, self.toast_kind = msg, time.time(), kind

    def set_mode(self, new_mode):
        if new_mode in self.MODES:
            self.mode = new_mode
            self.trigger_action(f"Mode -> {new_mode}")


state = StateManager()

# Every hand control that can be switched off / on (id, label, scope).
CONTROLS = [
    ("mode_toggle", "Fist hold: switch mode", "any"),
    ("lock", "Two fingers hold: lock / unlock", "any"),
    ("window_resize", "Shaka: window size", "any"),
    ("volume", "Index + thumb: volume", MODE_MEDIA),
    ("play_pause", "Open palm hold: play / pause", MODE_MEDIA),
    ("track_flick", "Hand flick: next / previous track", MODE_MEDIA),
    ("mute", "Quick fist tap: mute", MODE_MEDIA),
    ("index_draw", "Index finger: draw / erase / shapes", MODE_DRAWING),
    ("toolbar_hover", "Index on top bars: press buttons", MODE_DRAWING),
    ("pinch", "Pinch: finish shape", MODE_DRAWING),
    ("undo_redo", "Three fingers: undo / redo", MODE_DRAWING),
    ("color_flick", "Flick left / right: color", MODE_DRAWING),
    ("tool_flick", "Flick up / down: tool", MODE_DRAWING),
    ("brush_size", "Horns: brush size", MODE_DRAWING),
    ("clear", "Four fingers hold: clear", MODE_DRAWING),
    ("save", "Open palm at top: save", MODE_DRAWING),
]
enabled_controls = {cid: True for cid, _label, _scope in CONTROLS}
# gestures that map to exactly one control (shown as "(OFF)" in the status panel when disabled)
GESTURE_CONTROL = {"shaka": "window_resize", "horns": "brush_size", "three_finger": "undo_redo",
                   "pinch": "pinch", "two_finger": "lock"}


def enabled(cid):
    return enabled_controls.get(cid, True)


def set_control(cid, on):
    enabled_controls[cid] = bool(on)
    label = next(l for c, l, _ in CONTROLS if c == cid)
    msg = f"{label}: {'ON' if on else 'OFF'}"
    state.log(msg)
    state.notify(msg)


def enable_all():
    for cid in enabled_controls:
        enabled_controls[cid] = True
    state.log("All hand controls ON")
    state.notify("ALL HAND CONTROLS ON")


class HistoryManager:
    """Undo / redo stacks of full canvas snapshots."""

    def __init__(self, max_len=MAX_HISTORY):
        self.undo_stack, self.redo_stack, self.max_len = [], [], max_len

    def push(self, canvas):
        """Save the canvas state BEFORE an operation modifies it (invalidates redo)."""
        self.redo_stack.clear()
        if self.undo_stack and np.array_equal(self.undo_stack[-1], canvas):
            return
        self.undo_stack.append(canvas.copy())
        if len(self.undo_stack) > self.max_len:
            self.undo_stack.pop(0)

    def undo(self, current):
        if not self.undo_stack:
            return None
        self.redo_stack.append(current.copy())
        if len(self.redo_stack) > self.max_len:
            self.redo_stack.pop(0)
        return self.undo_stack.pop()

    def redo(self, current):
        if not self.redo_stack:
            return None
        self.undo_stack.append(current.copy())
        if len(self.undo_stack) > self.max_len:
            self.undo_stack.pop(0)
        return self.redo_stack.pop()

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


history = HistoryManager()


class ShapeRenderer:
    def __init__(self):
        self.shape_mode = None
        self.start_point = None
        self.current_point = None

    def start_shape(self, point, shape_type):
        self.shape_mode, self.start_point, self.current_point = shape_type, point, point

    def update_preview(self, point):
        self.current_point = point

    def _draw(self, img, color, thickness):
        x1, y1 = self.start_point
        x2, y2 = self.current_point
        if self.shape_mode == "line":
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        elif self.shape_mode == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        elif self.shape_mode == "circle":
            cv2.circle(img, (x1, y1), int(math.hypot(x2 - x1, y2 - y1)), color, thickness)

    def commit(self, canvas, color, thickness):
        if self.shape_mode and self.start_point and self.current_point:
            self._draw(canvas, color, thickness)
        self.cancel()
        return canvas

    def get_preview(self, canvas, color, thickness):
        if not (self.shape_mode and self.start_point and self.current_point):
            return None
        preview = canvas.copy()
        self._draw(preview, color, thickness)
        return preview

    def cancel(self):
        self.shape_mode = self.start_point = self.current_point = None


shape_renderer = ShapeRenderer()


# ============================================================================
# DWELL (hold-to-confirm, fires once per hold)
# ============================================================================
_dwell_start = {}
_dwell_fired = set()
_dwell_seen = set()


def dwell(name, secs):
    """Call every frame while the condition holds. True once when held long enough."""
    _dwell_seen.add(name)
    if name in _dwell_fired:
        return False
    now = time.time()
    start = _dwell_start.setdefault(name, now)
    if now - start >= secs:
        _dwell_fired.add(name)
        return True
    return False


def dwell_progress(name, secs):
    start = _dwell_start.get(name)
    if start is None or name in _dwell_fired:
        return 0.0
    return min(1.0, (time.time() - start) / secs)


def dwell_sweep():
    """Call once per frame: forget dwells whose condition was not seen."""
    global _dwell_fired
    for n in list(_dwell_start):
        if n not in _dwell_seen:
            _dwell_start.pop(n)
    _dwell_fired &= _dwell_seen
    _dwell_seen.clear()


# ============================================================================
# TOOLBAR  (finger-hover, mouse-click and gestures all trigger the same actions)
# ============================================================================
PANEL_KEYS = {"TOOLS": "tool", "COLORS": "color", "SIZES": "size"}
HEADER_BUTTONS = [("panel", "TOOLS"), ("panel", "COLORS"), ("panel", "SIZES"),
                  ("tool", "UNDO"), ("tool", "REDO"), ("tool", "CLEAR"), ("tool", "SAVE")]


class Toolbar:
    """Drawing mode: a slim header that is always visible (it shows the current tool / color /
    size and holds UNDO REDO CLEAR SAVE) plus three bars that open and close on demand:
    shapes/tools, colors, brush sizes.  Media mode: one row of media buttons."""

    def __init__(self, y_start=10, header_h=42, tool_h=44, color_h=40, size_h=30, gap=4):
        self.y_start, self.header_h, self.gap = y_start, header_h, gap
        self.tool_h, self.color_h, self.size_h = tool_h, color_h, size_h
        self.open = {"tool": False, "color": False, "size": False}   # all bars start closed
        self.hover = None      # (zone_type, name) the fingertip is aiming at

    def toggle(self, key):
        self.open[key] = not self.open[key]

    def rows(self, mode):
        """[(kind, y0, y1, [(zone_type, name), ...]), ...] for what is currently on screen."""
        y = self.y_start
        if mode == MODE_MEDIA:
            return [("media", y, y + self.header_h, [("media", n) for n in MEDIA_BUTTONS])]
        if mode != MODE_DRAWING:
            return []
        rows = [("header", y, y + self.header_h, HEADER_BUTTONS)]
        y += self.header_h
        bars = [("tool", self.tool_h, [("tool", n) for n in SHAPE_TOOLS]),
                ("color", self.color_h, [("color", n) for n in COLOR_ORDER]),
                ("size", self.size_h, [("size", n) for n in SIZES])]
        for key, height, items in bars:
            if self.open[key]:
                y += self.gap
                rows.append((key, y, y + height, items))
                y += height
        return rows

    def bottom(self, mode):
        rows = self.rows(mode)
        return rows[-1][2] if rows else 0

    def get_zone(self, x, y, frame_w, mode):
        for _kind, y0, y1, items in self.rows(mode):
            if y0 <= y <= y1:
                col_w = max(1, frame_w // len(items))
                return items[int(clamp(x // col_w, 0, len(items) - 1))]
        return None

    def render(self, frame, mode, active_tool, current_color, current_size):
        w = frame.shape[1]
        for _kind, y0, y1, items in self.rows(mode):
            n = len(items)
            col_w = w // n
            for i, (ztype, item) in enumerate(items):
                x1, x2 = i * col_w, i * col_w + col_w
                if ztype == "panel":
                    self._draw_panel_button(frame, x1, x2, y0, y1, item,
                                            active_tool, current_color, current_size)
                else:
                    if ztype == "color":
                        bg = COLORS[item]
                        active = item == current_color
                        border = (255, 255, 0) if active else (80, 80, 100)
                        label, label_color, scale = item, text_color_for(bg), 0.4
                    else:
                        active = (ztype == "tool" and item == active_tool) or \
                                 (ztype == "size" and item == current_size)
                        bg = (0, 100, 200) if active else (40, 40, 60)
                        border = (0, 150, 255) if active else (100, 100, 130)
                        label, label_color, scale = str(item), (255, 255, 255), 0.5
                    cv2.rectangle(frame, (x1, y0), (x2, y1), bg, -1)
                    cv2.rectangle(frame, (x1, y0), (x2, y1), border, 2 if active else 1)
                    put_centered(frame, label, x1, x2, (y0 + y1) // 2, scale, label_color)
                if self.hover == (ztype, item):
                    cv2.rectangle(frame, (x1, y0), (x2, y1), (0, 255, 100), 2)
                    p = dwell_progress(f"tb:{ztype}:{item}", HOVER_CONFIRM_TIME)
                    if p > 0:
                        cv2.rectangle(frame, (x1, y1 - 5), (x1 + int(p * col_w), y1), (0, 255, 100), -1)

    def _draw_panel_button(self, frame, x1, x2, y0, y1, item, active_tool, current_color, current_size):
        """Header toggle: shows the current value and a small arrow (down = closed, up = open)."""
        key = PANEL_KEYS[item]
        is_open = self.open[key]
        if key == "color":
            bg = COLORS[current_color]
            caption, value = "COLOR", current_color
            txt = text_color_for(bg)
        else:
            bg = (0, 100, 200) if is_open else (40, 40, 60)
            caption, value = ("TOOL", active_tool) if key == "tool" else ("SIZE", f"{current_size}px")
            txt = (255, 255, 255)
        border = (255, 255, 0) if (is_open and key == "color") else \
                 (0, 150, 255) if is_open else (100, 100, 130)
        cv2.rectangle(frame, (x1, y0), (x2, y1), bg, -1)
        cv2.rectangle(frame, (x1, y0), (x2, y1), border, 2 if is_open else 1)
        put_centered(frame, caption, x1, x2 - 14, y0 + 11, 0.34, txt)
        put_centered(frame, value, x1, x2 - 14, y1 - 13, 0.5, txt)
        ax, ay = x2 - 11, y0 + 12
        pts = [(ax - 5, ay + 3), (ax + 5, ay + 3), (ax, ay - 4)] if is_open else \
              [(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)]
        cv2.fillPoly(frame, [np.array(pts, np.int32)], txt)


toolbar = Toolbar()


# ============================================================================
# WINDOW MANAGER (resize / fullscreen, works with every HighGUI backend)
# ============================================================================
class WindowManager:
    def __init__(self, name, frame_w, frame_h):
        self.name, self.fw, self.fh = name, frame_w, frame_h
        self.scale = 1.0
        self.fullscreen = False
        self._applied = None
        self._last_apply = 0.0
        self._was_visible = False
        sw, sh = get_screen_size() or (1920, 1080)
        self.max_scale = max(1.0, min(WINDOW_SCALE_MAX, 0.95 * sw / frame_w, 0.88 * sh / frame_h))

    def create(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        self.apply(force=True)

    def set_scale(self, s):
        self.scale = clamp(s, WINDOW_SCALE_MIN, self.max_scale)
        self.apply()

    def apply(self, force=False):
        if self.fullscreen:
            return
        size = (int(self.fw * self.scale), int(self.fh * self.scale))
        now = time.time()
        if not force and (size == self._applied or now - self._last_apply < 0.05):
            return
        try:
            cv2.resizeWindow(self.name, size[0], size[1])
        except cv2.error:
            pass
        self._applied, self._last_apply = size, now

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        try:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        except cv2.error:
            pass
        if not self.fullscreen:
            self.apply(force=True)

    def closed(self):
        """True once the user has closed the window with its X button."""
        try:
            v = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return False
        if v >= 1:
            self._was_visible = True
            return False
        return self._was_visible


window = None


def draw_arrow(frame, cx, cy, up, color):
    d = -1 if up else 1
    pts = [(cx - 5, cy - 3 * d), (cx + 5, cy - 3 * d), (cx, cy + 4 * d)]
    cv2.fillPoly(frame, [np.array(pts, np.int32)], color)


class DashPanel:
    """Bottom status panel. Fold it with its arrow (or V), resize it by dragging its top edge
    (or < and >). Dragging it all the way down folds it, dragging it up opens it."""

    def __init__(self):
        self.open = True
        self.height = DASH_DEFAULT_H
        self.dragging = False
        self.frame_h = 480

    def max_height(self, frame_h):
        return int(min(DASH_MAX_H, frame_h - 150))

    def current_h(self, frame_h):
        self.frame_h = frame_h
        if not self.open:
            return DASH_STRIP_H
        return int(clamp(self.height, DASH_MIN_H, self.max_height(frame_h)))

    def top(self, frame_h):
        return frame_h - self.current_h(frame_h)

    def toggle(self):
        self.open = not self.open

    def resize(self, delta):
        self.open = True
        self.height = int(clamp(self.height + delta, DASH_MIN_H, self.max_height(self.frame_h)))

    def drag_to(self, y, frame_h):
        new_h = frame_h - y
        if new_h <= DASH_STRIP_H + 14:
            self.open = False
        else:
            self.open = True
            self.height = int(clamp(new_h, DASH_MIN_H, self.max_height(frame_h)))


dash = DashPanel()


class ControlsPanel:
    """Overlay listing the hand controls of the current mode with ON/OFF check boxes.
    Hand controls are paused while it is open so nothing gets drawn or switched by accident."""

    def __init__(self):
        self.open = False
        self.hit = []      # [(x1, y1, x2, y2, action, arg)] rebuilt on every render

    def toggle(self):
        self.open = not self.open

    def visible_items(self):
        view = state.mode if state.mode in (MODE_MEDIA, MODE_DRAWING) else state.resume_mode
        return view, [c for c in CONTROLS if c[2] in ("any", view)]

    def click(self, x, y):
        for x1, y1, x2, y2, action, arg in self.hit:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "item":
                    set_control(arg, not enabled(arg))
                elif action == "all_on":
                    enable_all()
                elif action == "close":
                    self.open = False
                return True
        return False

    def render(self, frame, top):
        self.hit = []
        if not self.open:
            return
        w = frame.shape[1]
        view, items = self.visible_items()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)
        title = f"HAND CONTROLS ({view} mode)  -  click to switch ON / OFF.  Gestures are paused while this is open."
        cv2.putText(frame, title, (10, 22), FONT, fit_scale([title], w - 20, 0.48), (0, 220, 255), 1, cv2.LINE_AA)

        per_col = (len(items) + 1) // 2
        btn_h = 26
        row_h = int(clamp((top - 40 - (btn_h + 14)) // max(1, per_col), 16, 30))
        col_w = w // 2
        box = 14
        for idx, (cid, label, _scope) in enumerate(items):
            col, row = divmod(idx, per_col)
            cx = 10 + col * col_w
            y1 = 36 + row * row_h
            on = enabled(cid)
            by = y1 + (row_h - box) // 2
            cv2.rectangle(frame, (cx, by), (cx + box, by + box), (60, 200, 60) if on else (50, 50, 70), -1)
            cv2.rectangle(frame, (cx, by), (cx + box, by + box), (200, 200, 200), 1)
            if on:
                cv2.line(frame, (cx + 3, by + 7), (cx + 6, by + 11), (0, 0, 0), 2)
                cv2.line(frame, (cx + 6, by + 11), (cx + 11, by + 3), (0, 0, 0), 2)
            text = label if on else f"{label}  (OFF)"
            tx = cx + box + 8
            sc = fit_scale([text], col_w - (tx - cx) - 16, 0.45)
            cv2.putText(frame, text, (tx, by + box - 2), FONT, sc,
                        (255, 255, 255) if on else (140, 140, 160), 1, cv2.LINE_AA)
            self.hit.append((cx - 4, y1, cx + col_w - 14, y1 + row_h - 1, "item", cid))

        by1 = top - btn_h - 8
        for bx1, bx2, label, action in ((10, 120, "ALL ON", "all_on"), (130, 240, "CLOSE", "close")):
            cv2.rectangle(frame, (bx1, by1), (bx2, by1 + btn_h), (40, 40, 60), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by1 + btn_h), (100, 100, 130), 1)
            put_centered(frame, label, bx1, bx2, by1 + btn_h // 2, 0.5, (255, 255, 255))
            self.hit.append((bx1, by1, bx2, by1 + btn_h, action, None))


controls_panel = ControlsPanel()


# ============================================================================
# GESTURE HELPERS
# ============================================================================
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def px_dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def palm_size(lm, w, h):
    return max(1.0, px_dist(lm[0], lm[9], w, h))


def _ext(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def _fingers(lm):
    """(index, middle, ring, pinky) extended?"""
    return (_ext(lm, 8, 6), _ext(lm, 12, 10), _ext(lm, 16, 14), _ext(lm, 20, 18))


def thumb_extended(lm):
    """Heuristic: thumb tip farther from pinky base than the thumb IP joint."""
    return abs(lm[4].x - lm[17].x) > abs(lm[3].x - lm[17].x)


def _four_up(lm):
    return all(_fingers(lm))


def is_open_palm(lm):
    return _four_up(lm) and thumb_extended(lm)


def is_four_fingers(lm):
    return _four_up(lm) and not thumb_extended(lm)


def is_closed_fist(lm):
    return _fingers(lm) == (False, False, False, False)


def is_two_finger(lm):
    return _fingers(lm) == (True, True, False, False)


def is_three_fingers(lm):
    return _fingers(lm) == (True, True, True, False)


def is_index_point(lm):
    return _fingers(lm) == (True, False, False, False)


def is_horns(lm):
    """Index + pinky up, middle + ring down (thumb ignored)."""
    return _fingers(lm) == (True, False, False, True)


def is_shaka(lm):
    """Only pinky up and thumb out."""
    return _fingers(lm) == (False, False, False, True) and thumb_extended(lm)


def detect_pinching(lm):
    return calc_distance(lm[4], lm[8]) < 0.05


def detect_flick(hist):
    """'left' | 'right' | 'up' | 'down' | None from the last 5 wrist positions (mirrored frame)."""
    if len(hist) < hist.maxlen:
        return None
    dx = hist[-1][0] - hist[0][0]
    dy = hist[-1][1] - hist[0][1]
    if abs(dx) >= SWIPE_THRESHOLD and abs(dx) > 1.5 * abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) >= SWIPE_THRESHOLD_V and abs(dy) > 1.5 * abs(dx):
        return "down" if dy > 0 else "up"
    return None


def hand_still(hist):
    if len(hist) < hist.maxlen:
        return False
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    return (max(xs) - min(xs)) < STILL_THRESHOLD and (max(ys) - min(ys)) < STILL_THRESHOLD


def volume_from_distance(dist):
    norm = max(0.0, min(1.0, (dist - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)))
    return int(round(norm * 100))


def _classify_gesture(lm, mode):
    """Pure classifier: returns (type, description). No side effects."""
    if is_closed_fist(lm):
        return "fist", "Closed Fist"
    if is_shaka(lm):
        return "shaka", "Shaka (window size)"
    if is_horns(lm):
        return "horns", "Horns (brush size)"
    if is_two_finger(lm):
        return "two_finger", "Two Fingers"
    if is_three_fingers(lm):
        return "three_finger", "Three Fingers"
    if is_four_fingers(lm):
        return "four_finger", "Four Fingers"
    if is_open_palm(lm):
        return "open_palm", "Open Palm"
    if mode == MODE_DRAWING and detect_pinching(lm):
        return "pinch", "Pinch"
    if is_index_point(lm):
        return "index_point", "Index Point"
    return "none", "Idle"


# Only gestures that do something in the current mode are reported; the rest read as "Idle".
MODE_GESTURES = {
    MODE_LOCKED: {"two_finger"},
    MODE_MEDIA: {"fist", "shaka", "two_finger", "four_finger", "open_palm", "index_point"},
    MODE_DRAWING: {"fist", "shaka", "two_finger", "three_finger", "four_finger", "open_palm",
                   "pinch", "index_point", "horns"},
}


def recognize_gesture(lm, mode):
    gtype, desc = _classify_gesture(lm, mode)
    if gtype != "none" and gtype not in MODE_GESTURES.get(mode, ()):
        return "none", "Idle"
    return gtype, desc


# ============================================================================
# GLOBAL DRAWING STATE
# ============================================================================
canvas = None
active_tool = "DRAW"
current_color = "RED"
brush_size = 10
last_point = None
fist_start = None
show_window_hint = False


class Cursor:
    def __init__(self):
        self.pos = None
        self.prev_smooth = None
        self.history = deque(maxlen=5)

    def update(self, x, y):
        self.history.append((x, y))
        rx = sum(p[0] for p in self.history) / len(self.history)
        ry = sum(p[1] for p in self.history) / len(self.history)
        if self.prev_smooth is not None:
            sx = SMOOTHING_FACTOR * rx + (1 - SMOOTHING_FACTOR) * self.prev_smooth[0]
            sy = SMOOTHING_FACTOR * ry + (1 - SMOOTHING_FACTOR) * self.prev_smooth[1]
        else:
            sx, sy = rx, ry
        self.prev_smooth = (sx, sy)
        self.pos = (int(sx), int(sy))
        return self.pos

    def reset(self):
        self.pos = self.prev_smooth = None
        self.history.clear()


cursor = Cursor()
size_smoother = Smoother(0.3)
scale_smoother = Smoother(0.25)


# ============================================================================
# ACTIONS  (shared by gestures, finger-hover, mouse clicks and keyboard)
# ============================================================================
def _do_undo():
    prev = history.undo(canvas)
    if prev is None:
        state.trigger_action("Undo - nothing to undo", "draw")
        state.notify("NOTHING TO UNDO", "draw")
    else:
        canvas[:] = prev
        shape_renderer.cancel()
        state.trigger_action("Undo successful", "draw")
        state.notify("UNDO", "draw")


def _do_redo():
    nxt = history.redo(canvas)
    if nxt is None:
        state.trigger_action("Redo - nothing to redo", "draw")
        state.notify("NOTHING TO REDO", "draw")
    else:
        canvas[:] = nxt
        shape_renderer.cancel()
        state.trigger_action("Redo successful", "draw")
        state.notify("REDO", "draw")


def _do_clear():
    history.push(canvas)
    canvas[:] = CANVAS_BACKGROUND
    shape_renderer.cancel()
    state.trigger_action("Canvas cleared", "draw")
    state.notify("CANVAS CLEARED", "draw")


def _do_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    if cv2.imwrite(path, canvas):
        state.notify(f"SAVED: {os.path.basename(path)}", "draw")
        state.trigger_action(f"Saved -> {path}", "draw")
    else:
        state.notify("ERROR saving canvas", "draw")
        state.trigger_action("Save failed", "draw")


def set_tool(name):
    global active_tool, last_point
    active_tool = name
    last_point = None
    shape_renderer.cancel()
    state.log(f"Tool -> {name}", "draw")
    state.notify(f"TOOL: {name}", "draw")


def set_color(name):
    global current_color
    current_color = name
    state.log(f"Color -> {name}", "draw")
    state.notify(f"COLOR: {name}", "draw")


def set_size(px):
    global brush_size
    brush_size = int(px)
    state.log(f"Size -> {brush_size}px", "draw")
    state.notify(f"SIZE: {brush_size}px", "draw")


def cycle_color(step):
    set_color(COLOR_ORDER[(COLOR_ORDER.index(current_color) + step) % len(COLOR_ORDER)])


def cycle_tool(step):
    set_tool(SHAPE_TOOLS[(SHAPE_TOOLS.index(active_tool) + step) % len(SHAPE_TOOLS)])


def step_size(direction):
    if direction > 0:
        bigger = [s for s in SIZES if s > brush_size]
        set_size(bigger[0] if bigger else SIZES[-1])
    else:
        smaller = [s for s in SIZES if s < brush_size]
        set_size(smaller[-1] if smaller else SIZES[0])


def toggle_mode():
    shape_renderer.cancel()
    state.set_mode(MODE_DRAWING if state.mode == MODE_MEDIA else MODE_MEDIA)


def toggle_lock():
    global last_point
    if state.system_locked:
        state.system_locked = False
        state.set_mode(state.resume_mode)
        state.trigger_action("System UNLOCKED")
        state.show_help = False
    else:
        state.resume_mode = state.mode
        state.system_locked = True
        last_point = None
        shape_renderer.cancel()
        state.set_mode(MODE_LOCKED)
        state.trigger_action("System LOCKED")


def _set_volume(pct):
    pct = int(clamp(pct, 0, 100))
    if media.set_volume(pct):
        state.log(f"Volume -> {pct}%", "media")
    else:
        state.log("Volume control failed", "media")
    state.volume = pct


def _media_action(name):
    simple = {"PREV": (media.previous_track, "Previous Track"),
              "PLAY/PAUSE": (media.play_pause, "Play/Pause toggled"),
              "NEXT": (media.next_track, "Next Track"),
              "MUTE": (media.mute, "Mute toggled")}
    if name in simple:
        fn, label = simple[name]
        state.trigger_action(label if fn() else f"{label} - failed", "media")
    elif name == "VOL+":
        _set_volume(state.volume + 5)
    elif name == "VOL-":
        _set_volume(state.volume - 5)


def activate(zone_type, name):
    """Run the action behind a toolbar button."""
    if zone_type == "tool":
        if name in SHAPE_TOOLS:
            set_tool(name)
        elif name == "UNDO":
            _do_undo()
        elif name == "REDO":
            _do_redo()
        elif name == "CLEAR":
            _do_clear()
        elif name == "SAVE":
            _do_save()
    elif zone_type == "color":
        set_color(name)
    elif zone_type == "size":
        set_size(name)
    elif zone_type == "panel":
        key = PANEL_KEYS[name]
        toolbar.toggle(key)
        state.log(f"{name.title()} bar {'opened' if toolbar.open[key] else 'closed'}", "draw")
    elif zone_type == "media":
        _media_action(name)


# ============================================================================
# GESTURE HANDLERS
# ============================================================================
def _handle_lock_unlock():
    if dwell("lock", CONFIRMATION_TIME):
        toggle_lock()


def _handle_fist(is_fist):
    """Hold >= CONFIRMATION_TIME -> toggle mode. Quick tap in MEDIA -> mute."""
    global fist_start
    now = time.time()
    if is_fist:
        if fist_start is None:
            fist_start = now
        if enabled("mode_toggle") and dwell("fist_toggle", CONFIRMATION_TIME):
            toggle_mode()
    elif fist_start is not None:
        held = now - fist_start
        fist_start = None
        if state.mode == MODE_MEDIA and enabled("mute") and 0.1 <= held < CONFIRMATION_TIME \
                and not state.in_cooldown():
            _media_action("MUTE")


def _handle_resize(lm, w, h):
    """Shaka: thumb-tip <-> pinky-tip span (normalised by palm size) sets the window size."""
    if window is None:
        return
    ratio = px_dist(lm[4], lm[20], w, h) / palm_size(lm, w, h)
    t = clamp((ratio - RESIZE_RATIO_MIN) / (RESIZE_RATIO_MAX - RESIZE_RATIO_MIN), 0.0, 1.0)
    target = WINDOW_SCALE_MIN + t * (window.max_scale - WINDOW_SCALE_MIN)
    window.set_scale(scale_smoother.update(target))
    state.notify(f"WINDOW SIZE: {window.scale:.1f}x")


def _handle_brush_size(lm, w, h):
    """Horns: index-tip <-> pinky-tip span (normalised by palm size) sets the brush size."""
    global brush_size
    ratio = px_dist(lm[8], lm[20], w, h) / palm_size(lm, w, h)
    t = clamp((ratio - BRUSH_RATIO_MIN) / (BRUSH_RATIO_MAX - BRUSH_RATIO_MIN), 0.0, 1.0)
    target = MIN_BRUSH_SIZE + t * (MAX_BRUSH_SIZE - MIN_BRUSH_SIZE)
    brush_size = int(round(size_smoother.update(target)))
    state.notify(f"BRUSH SIZE: {brush_size}px", "draw")


def _handle_media(gesture_type, lm, w):
    """MEDIA mode gestures: flick, play/pause, volume."""
    flick = detect_flick(state.swipe_history) \
        if gesture_type in ("open_palm", "four_finger") and enabled("track_flick") else None
    if flick in ("left", "right") and not state.in_cooldown():
        state.swipe_history.clear()
        if flick == "right":
            ok = media.next_track()
            state.trigger_action("Right Swipe -> Next Track" if ok else "Next Track failed", "media")
            return "swipe_right", "Right Swipe"
        ok = media.previous_track()
        state.trigger_action("Left Swipe -> Previous Track" if ok else "Previous Track failed", "media")
        return "swipe_left", "Left Swipe"

    if gesture_type == "open_palm" and enabled("play_pause") and not state.in_cooldown() \
            and hand_still(state.swipe_history):
        if dwell("palm_pp", PALM_DWELL):
            _media_action("PLAY/PAUSE")

    if gesture_type == "index_point" and enabled("volume"):
        dist = calc_distance(lm[4], lm[8])
        new_volume = volume_from_distance(dist)
        now = time.time()
        if abs(new_volume - state.volume) >= 3 and now - state.last_volume_time >= VOLUME_MIN_INTERVAL:
            state.last_volume_time = now
            _set_volume(new_volume)
        return "volume", f"Vol {state.volume}% (d={dist:.2f})"
    return None


def _handle_toolbar(cur_pos, w):
    zone = toolbar.get_zone(cur_pos[0], cur_pos[1], w, state.mode)
    if zone is None:
        return
    if dwell(f"tb:{zone[0]}:{zone[1]}", HOVER_CONFIRM_TIME):
        activate(*zone)
        state.draw_block_until = time.time() + 0.5   # the bars may have moved under the fingertip


def _handle_drawing(cur_pos):
    global last_point
    color = COLORS[current_color]
    if active_tool == "DRAW":
        if last_point is None:
            history.push(canvas)
            last_point = cur_pos
        cv2.line(canvas, last_point, cur_pos, color, brush_size)
        last_point = cur_pos
    elif active_tool == "ERASE":
        if last_point is None:
            history.push(canvas)
            last_point = cur_pos
        cv2.line(canvas, last_point, cur_pos, CANVAS_BACKGROUND, ERASER_SIZE)
        last_point = cur_pos
    elif active_tool in ("LINE", "RECT", "CIRCLE"):
        if shape_renderer.shape_mode is None:
            shape_renderer.start_shape(cur_pos, active_tool.lower())
            state.log(f"Start {active_tool} at {cur_pos}", "draw")
        else:
            shape_renderer.update_preview(cur_pos)


def _handle_pinch():
    if shape_renderer.shape_mode and not state.in_cooldown():
        history.push(canvas)
        shape_renderer.commit(canvas, COLORS[current_color], brush_size)
        state.trigger_action(f"{active_tool} committed", "draw")


def _handle_drawing_mode(gesture_type, cur_pos, lm, w, h):
    global last_point
    if gesture_type != "index_point":
        last_point = None

    ready = not state.in_cooldown()
    still = hand_still(state.swipe_history)
    flick = detect_flick(state.swipe_history)

    if gesture_type == "index_point" and cur_pos:
        if cur_pos[1] <= toolbar.bottom(MODE_DRAWING):
            last_point = None
            if enabled("toolbar_hover"):
                _handle_toolbar(cur_pos, w)
        elif time.time() < state.draw_block_until or not enabled("index_draw"):
            last_point = None
        else:
            _handle_drawing(cur_pos)
    elif gesture_type == "pinch":
        if enabled("pinch"):
            _handle_pinch()
    elif gesture_type == "three_finger" and enabled("undo_redo"):
        if flick in ("left", "right") and ready:
            state.swipe_history.clear()
            if flick == "right":
                _do_redo()
            else:
                _do_undo()
        elif still and ready and dwell("undo_hold", UNDO_HOLD):
            _do_undo()
    elif gesture_type in ("open_palm", "four_finger"):
        horizontal = flick in ("left", "right")
        if flick and ready and enabled("color_flick" if horizontal else "tool_flick"):
            state.swipe_history.clear()
            if horizontal:
                cycle_color(1 if flick == "right" else -1)
            else:
                cycle_tool(1 if flick == "down" else -1)
            state.last_action_time = time.time()
        elif gesture_type == "four_finger":
            if enabled("clear") and still and ready and dwell("clear", CONFIRMATION_TIME):
                _do_clear()
        elif cur_pos and cur_pos[1] < h // 3:
            if enabled("save") and still and ready and dwell("save", CONFIRMATION_TIME):
                _do_save()
    elif gesture_type == "horns" and enabled("brush_size"):
        _handle_brush_size(lm, w, h)


# ============================================================================
# RENDERING
# ============================================================================
HELP_LINES = {
    MODE_LOCKED: [
        "SYSTEM LOCKED",
        "Hold two fingers (index + middle) for half a second to unlock.",
        "Keys: L unlock, Q quit.",
    ],
    MODE_MEDIA: [
        "MEDIA MODE  (H hides this help)",
        "Index finger + thumb: distance = volume",
        "Open palm (hold still): play / pause",
        "Open hand flick right / left: next / previous track",
        "Quick fist tap: mute / unmute",
        "Fist (hold): switch to DRAWING mode",
        "Mouse: click the buttons at the top",
        "Keys: P play/pause  N next  B previous  X mute  [ ] volume",
    ],
    MODE_DRAWING: [
        "DRAWING MODE  (H hides this help)",
        "Index finger: draw / aim at a toolbar button (hold 0.4 s to press)",
        "Top bar: TOOL / COLOR / SIZE open and close their bars (keys 1 2 3)",
        "Pinch: finish line / rect / circle",
        "Three fingers: hold or flick left = undo, flick right = redo",
        "Open hand flick: left/right = color, up/down = tool",
        "Horns (index + pinky): spread fingers = brush size",
        "Four fingers (hold): clear   |   Open palm at top (hold): save",
        "Fist (hold): switch to MEDIA mode",
        "Keys: U undo R redo C clear S save D draw E erase T tool , . color [ ] size 1 2 3 bars",
    ],
}
COMMON_HELP = [
    "Any mode: shaka (thumb + pinky) spread = window size, two fingers (hold) = lock",
    "Keys: M mode  L lock  F fullscreen  + / - window  H help  Q quit",
    "Status panel: drag its top edge to resize, click the arrow (or V) to fold. CONTROLS (G): turn hand controls on/off.",
]
PROGRESS_BARS = [
    ("clear", "HOLD TO CLEAR CANVAS", CONFIRMATION_TIME, (255, 100, 0)),
    ("save", "HOLD TO SAVE", CONFIRMATION_TIME, (0, 200, 0)),
    ("lock", "HOLD TO LOCK / UNLOCK", CONFIRMATION_TIME, (0, 200, 255)),
    ("fist_toggle", "SWITCHING MODE...", CONFIRMATION_TIME, (255, 200, 50)),
    ("undo_hold", "UNDO...", UNDO_HOLD, (255, 0, 255)),
]


def render_help(frame, top):
    w = frame.shape[1]
    lines = HELP_LINES[state.mode] + (COMMON_HELP if state.mode != MODE_LOCKED else [])
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    step = clamp((top - 34) / max(1, len(lines)), 14, 24)
    scale = min(fit_scale(lines, w - 24, 0.5), 0.5 * step / 22)
    y = 26.0
    for i, line in enumerate(lines):
        if y > top - 6:
            break
        cv2.putText(frame, line, (12, int(y)), FONT, scale,
                    (0, 220, 255) if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
        y += step


def render_progress(frame, top):
    w = frame.shape[1]
    y = top - 14
    for name, label, secs, color in PROGRESS_BARS:
        p = dwell_progress(name, secs)
        if p <= 0:
            continue
        bx = w // 2 - 100
        cv2.rectangle(frame, (bx, y), (bx + 200, y + 12), (40, 40, 60), -1)
        cv2.rectangle(frame, (bx, y), (bx + int(p * 200), y + 12), color, -1)
        put_outlined(frame, label, (bx, y - 5), 0.5, (255, 255, 255), 1)
        y -= 32


def render_media_extras(frame, y_top):
    w = frame.shape[1]
    y = y_top + 14
    bar_w = w - 30
    cv2.rectangle(frame, (15, y), (15 + bar_w, y + 16), (40, 40, 60), -1)
    cv2.rectangle(frame, (15, y), (15 + int(bar_w * state.volume / 100), y + 16), (255, 180, 0), -1)
    cv2.rectangle(frame, (15, y), (15 + bar_w, y + 16), (150, 150, 180), 1)
    put_outlined(frame, f"VOLUME {state.volume}%", (15, y + 42), 0.6, (255, 180, 0))


KIND_OF_MODE = {MODE_MEDIA: "media", MODE_DRAWING: "draw"}


def render_dashboard(frame, gesture_type, gesture_desc, fps):
    """Foldable / resizable status panel. Common status on the left; the middle and right
    columns and the log depend on the mode. Rows that do not fit the current height are hidden."""
    h, w = frame.shape[:2]
    ph = dash.current_h(h)
    top = h - ph
    mode = state.mode
    cv2.rectangle(frame, (0, top), (w, h), (10, 10, 25), -1)

    # ---- strip: fold arrow, title, CONTROLS button, drag grip ----
    cv2.rectangle(frame, (0, top), (w, top + DASH_STRIP_H), (28, 28, 52), -1)
    cv2.rectangle(frame, (0, top), (w, top + 2), (0, 200, 255) if dash.dragging else (60, 60, 100), -1)
    for dx in (-8, 0, 8):
        cv2.circle(frame, (w // 2 + dx, top + 5), 1, (150, 150, 190), -1)
    draw_arrow(frame, 14, top + 14, not dash.open, (220, 220, 255))
    cv2.putText(frame, "STATUS", (28, top + 19), FONT, 0.42, (220, 220, 255), 1, cv2.LINE_AA)
    if not dash.open:
        cv2.putText(frame, f"{mode} | {gesture_desc.upper()} | {fps:.0f} FPS", (100, top + 19), FONT, 0.42,
                    (0, 255, 0) if gesture_type != "none" else (150, 150, 150), 1, cv2.LINE_AA)
    n_off = sum(1 for v in enabled_controls.values() if not v)
    bx1, bx2 = w - CONTROLS_BTN_W, w - 6
    cv2.rectangle(frame, (bx1, top + 3), (bx2, top + DASH_STRIP_H - 3),
                  (0, 100, 200) if controls_panel.open else (40, 40, 60), -1)
    cv2.rectangle(frame, (bx1, top + 3), (bx2, top + DASH_STRIP_H - 3), (100, 100, 130), 1)
    put_centered(frame, "CONTROLS" if not n_off else f"CONTROLS ({n_off} OFF)", bx1, bx2,
                 top + DASH_STRIP_H // 2, 0.42, (255, 255, 255))
    if not dash.open:
        return

    # ---- body ----
    def text(rel_y, msg, x, color, scale=0.48):
        if rel_y + 4 <= ph:
            cv2.putText(frame, msg, (x, top + rel_y), FONT, scale, color, 1, cv2.LINE_AA)

    c2, c3 = int(w * 0.42), int(w * 0.70)
    rows = [84, 102, 120, 138]

    def put(msg, x, row, color):
        text(rows[row], msg, x, color)

    text(44, "SOUTHVILLE INTERNATIONAL SCHOOL AND COLLEGES", 15, (200, 200, 255), 0.5)
    text(60, "ADVANCED OPENCV LABORATORY  |  VARIANT 9", 15, (180, 180, 220), 0.45)

    locked = state.system_locked
    mode_color = (50, 200, 255) if mode == MODE_DRAWING else (255, 200, 50)
    ges_color = (0, 255, 0) if gesture_type != "none" else (150, 150, 150)
    put(f"MODE:    {mode}", 15, 0, mode_color)
    put(f"SYSTEM:  {'LOCKED' if locked else 'UNLOCKED'}", 15, 1, (255, 50, 50) if locked else (50, 255, 100))
    put(f"GESTURE: {gesture_desc.upper()}", 15, 2, ges_color)
    put(f"STATUS:  {'DETECTED' if gesture_type != 'none' else 'WAITING'}", 15, 3, (200, 200, 255))

    ready = not state.in_cooldown()
    cooldown = (f"COOLDOWN: {'READY' if ready else 'ACTIVE'}", (0, 255, 0) if ready else (255, 80, 80))
    win = f"WINDOW: {window.scale if window else 1.0:.1f}x"

    if mode == MODE_DRAWING:
        put(f"TOOL:    {active_tool}", c2, 0, (255, 255, 100))
        put(f"COLOR:   {current_color}", c2, 1, COLORS[current_color] if current_color != "BLACK" else (200, 200, 200))
        put(f"SIZE:    {brush_size}px", c2, 2, (200, 200, 255))
        put(cooldown[0], c3, 0, cooldown[1])
        put(f"FPS: {fps:.1f}", c3, 1, (0, 255, 0))
        put(win, c3, 2, (200, 200, 255))
        put(f"UNDO {len(history.undo_stack)} | REDO {len(history.redo_stack)}", c3, 3, (200, 200, 255))
    elif mode == MODE_MEDIA:
        put(f"VOLUME:  {state.volume}%", c2, 0, (255, 180, 0))
        put(cooldown[0], c3, 0, cooldown[1])
        put(f"FPS: {fps:.1f}", c3, 1, (0, 255, 0))
        put(win, c3, 2, (200, 200, 255))
    else:
        put(f"FPS: {fps:.1f}", c3, 0, (0, 255, 0))

    allowed = ("sys", KIND_OF_MODE.get(mode, "sys"))
    shown = [e for e in state.action_history if e[2] in allowed][-4:]
    n_fit = max(0, min(4, (ph - 4 - 156) // 12 + 1))
    if n_fit:
        for i, (ts, act, _kind) in enumerate(shown[-n_fit:]):
            text(156 + i * 12, f"[{ts}] {act}", 15, (160, 160, 200), 0.36)


def process_frame(frame, results, fps):
    """Recognise gestures, dispatch actions, compose the display frame."""
    global last_point, fist_start
    h, w = frame.shape[:2]

    gesture_type, gesture_desc = "none", "No Hand"
    cur_pos = None
    hand = None
    lm = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        gesture_type, gesture_desc = recognize_gesture(lm, state.mode)
        cid = GESTURE_CONTROL.get(gesture_type)
        if gesture_type == "two_finger" and state.system_locked:
            cid = None                      # unlocking always works
        if cid and not enabled(cid):
            gesture_desc = gesture_desc.split(" (")[0] + " (OFF)"
        cur_pos = cursor.update(int(lm[8].x * w), int(lm[8].y * h))
        state.swipe_history.append((lm[0].x * w, lm[0].y * h))   # frame is already mirrored
        if gesture_type != "horns":
            size_smoother.reset()
        if gesture_type != "shaka":
            scale_smoother.reset()

        live = not controls_panel.open      # hand controls pause while the CONTROLS panel is open
        _handle_fist(gesture_type == "fist" and not state.system_locked and live)

        if gesture_type == "two_finger" and live and (enabled("lock") or state.system_locked):
            _handle_lock_unlock()

        if not state.system_locked and live:
            if gesture_type == "shaka":
                if enabled("window_resize"):
                    _handle_resize(lm, w, h)
            elif state.mode == MODE_MEDIA:
                res = _handle_media(gesture_type, lm, w)
                if res:
                    gesture_type, gesture_desc = res
            elif state.mode == MODE_DRAWING:
                _handle_drawing_mode(gesture_type, cur_pos, lm, w, h)
    else:
        cursor.reset()
        last_point = None
        fist_start = None
        state.swipe_history.clear()
        size_smoother.reset()
        scale_smoother.reset()

    toolbar.hover = None
    if state.mode == MODE_DRAWING and not state.system_locked and not controls_panel.open \
            and enabled("toolbar_hover") and gesture_type == "index_point" and cur_pos and cur_pos[1] <= toolbar.bottom(MODE_DRAWING):
        toolbar.hover = toolbar.get_zone(cur_pos[0], cur_pos[1], w, MODE_DRAWING)

    dwell_sweep()

    # ---- compose --------------------------------------------------------
    if state.mode == MODE_DRAWING:
        preview = shape_renderer.get_preview(canvas, COLORS[current_color], brush_size)
        display = cv2.addWeighted(frame, 0.35, preview if preview is not None else canvas, 0.65, 0)
    elif state.mode == MODE_MEDIA:
        display = frame.copy()
    else:
        display = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

    if hand is not None:
        mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)
        if state.mode == MODE_DRAWING and cur_pos:          # fingertip cursor = drawing only
            color_map = {"index_point": (0, 255, 0), "pinch": (255, 255, 0),
                         "three_finger": (255, 0, 255), "four_finger": (0, 255, 255),
                         "horns": (0, 128, 255), "shaka": (255, 128, 0)}
            c = color_map.get(gesture_type, (0, 200, 255))
            cv2.circle(display, cur_pos, 12, c, 2)
            cv2.circle(display, cur_pos, 4, c, -1)
        elif state.mode == MODE_MEDIA and gesture_type == "volume":   # thumb-index volume span
            p1 = (int(lm[4].x * w), int(lm[4].y * h))
            p2 = (int(lm[8].x * w), int(lm[8].y * h))
            cv2.line(display, p1, p2, (255, 180, 0), 3)
            cv2.circle(display, p1, 7, (255, 180, 0), -1)
            cv2.circle(display, p2, 7, (255, 180, 0), -1)

    top = dash.top(h)
    toolbar.render(display, state.mode, active_tool, current_color, brush_size)
    if state.mode == MODE_MEDIA:
        render_media_extras(display, toolbar.bottom(MODE_MEDIA))

    if state.mode == MODE_DRAWING and gesture_type == "horns":     # live brush preview
        cx, cy = w // 2, toolbar.bottom(MODE_DRAWING) + 45
        cv2.circle(display, (cx, cy), max(1, brush_size // 2) + 2, (0, 0, 0), -1)
        cv2.circle(display, (cx, cy), max(1, brush_size // 2), COLORS[current_color], -1)

    if state.toast and time.time() - state.toast_time < 2.0 \
            and state.toast_kind in ("sys", KIND_OF_MODE.get(state.mode, "sys")):
        toast_y = toolbar.bottom(state.mode) + (95 if state.mode == MODE_MEDIA else 28)   # clear of the volume bar
        put_outlined(display, state.toast, (12, toast_y), 0.6, (0, 255, 100))

    render_progress(display, top)
    if state.show_help and not controls_panel.open:
        render_help(display, top)
    controls_panel.render(display, top)
    render_dashboard(display, gesture_type, gesture_desc, fps)

    if state.system_locked:
        put_outlined(display, "SYSTEM LOCKED", (w // 2 - 110, top // 2), 1.0, (255, 50, 50))
    return display


# ============================================================================
# CAMERA / INPUT / MAIN
# ============================================================================
def initialize_camera(index):
    if IS_WINDOWS:
        backends = [getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY]
    elif IS_MAC:
        backends = [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
    else:
        backends = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
    candidates = [(index, b) for b in backends]
    if IS_LINUX:
        candidates.insert(1, (f"/dev/video{index}", None))

    for src, backend in candidates:
        cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not IS_MAC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Camera device: {src} (backend {backend})")
            print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
            return cap
        cap.release()
    raise RuntimeError(f"Unable to open webcam (index {index}). "
                       "Check that no other app uses it, that camera permission is granted, "
                       "or try --camera 1")


def handle_mouse(event, x, y, w, h):
    """Mouse: drag the status panel's top edge, fold it, open CONTROLS, press toolbar buttons."""
    top = dash.top(h)
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(y - top) <= DRAG_GRAB:
            dash.dragging = True
            return
        if y >= top:                                   # inside the status panel
            if y <= top + DASH_STRIP_H:
                if x >= w - CONTROLS_BTN_W:
                    controls_panel.toggle()
                elif x <= 130:
                    dash.toggle()
            return
        if controls_panel.open:
            controls_panel.click(x, y)
            return
        if not state.system_locked:
            zone = toolbar.get_zone(x, y, w, state.mode)
            if zone:
                activate(*zone)
    elif event == cv2.EVENT_MOUSEMOVE and dash.dragging:
        dash.drag_to(y, h)
    elif event == cv2.EVENT_LBUTTONUP:
        dash.dragging = False


def handle_key(key):
    """Keyboard shortcuts. Returns True when the app should quit."""
    if key == 255 or key < 0:
        return False
    if key in (ord("q"), 27):
        return True
    ch = chr(key).lower()

    if ch == "f":
        window.toggle_fullscreen()
    elif ch in ("+", "="):
        window.set_scale(window.scale + 0.1)
    elif ch in ("-", "_"):
        window.set_scale(window.scale - 0.1)
    elif ch == "h":
        state.show_help = not state.show_help
    elif ch == "v":
        dash.toggle()
    elif ch == "<":
        dash.resize(-20)
    elif ch == ">":
        dash.resize(20)
    elif ch == "g":
        controls_panel.toggle()
    elif ch == "a" and controls_panel.open:
        enable_all()
    elif ch == "l":
        toggle_lock()
    elif state.system_locked:
        pass
    elif ch == "m":
        toggle_mode()
    elif state.mode == MODE_MEDIA:
        if ch == "p":
            _media_action("PLAY/PAUSE")
        elif ch == "n":
            _media_action("NEXT")
        elif ch == "b":
            _media_action("PREV")
        elif ch == "x":
            _media_action("MUTE")
        elif ch == "]":
            _media_action("VOL+")
        elif ch == "[":
            _media_action("VOL-")
    elif state.mode == MODE_DRAWING:
        if ch in ("1", "2", "3"):
            activate("panel", {"1": "TOOLS", "2": "COLORS", "3": "SIZES"}[ch])
        elif ch == "u":
            _do_undo()
        elif ch == "r":
            _do_redo()
        elif ch == "c":
            _do_clear()
        elif ch == "s":
            _do_save()
        elif ch == "d":
            set_tool("DRAW")
        elif ch == "e":
            set_tool("ERASE")
        elif ch == "t":
            cycle_tool(1)
        elif ch == ",":
            cycle_color(-1)
        elif ch == ".":
            cycle_color(1)
        elif ch == "]":
            step_size(1)
        elif ch == "[":
            step_size(-1)
    return False


def main():
    global canvas, window

    parser = argparse.ArgumentParser(description="Gesture-controlled drawing + media studio")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="webcam index (default 0)")
    parser.add_argument("--player", default=MEDIA_PLAYER,
                        help="Linux: playerctl player name to prefer (default %(default)s, '' = any)")
    args = parser.parse_args()
    media.player = args.player or None

    try:
        cap = initialize_camera(args.camera)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)

    vol = media.get_volume()
    if vol is not None:
        state.volume = vol

    mouse_events = deque(maxlen=200)

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE):
            mouse_events.append((event, x, y))

    fps_counter, fps_time, fps_display = 0, time.time(), 0
    failures = 0
    print("Keyboard: q/Esc quit | m mode | l lock | f fullscreen | + - window | h help")
    print("System starts LOCKED: hold two fingers (or press L) to unlock.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            failures += 1
            if failures >= 30:
                print("ERROR: webcam frame acquisition failed repeatedly.")
                break
            time.sleep(0.01)
            continue
        failures = 0
        frame = cv2.flip(frame, 1)
        ch, cw = frame.shape[:2]

        if canvas is None or canvas.shape[:2] != (ch, cw):
            canvas = np.full((ch, cw, 3), CANVAS_BACKGROUND, dtype=np.uint8)
            history.clear()
        if window is None:
            window = WindowManager(WINDOW_NAME, cw, ch)
            window.create()
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display, fps_counter, fps_time = fps_counter, 0, now

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display = process_frame(frame, results, fps_display)

        while mouse_events:
            ev, x, y = mouse_events.popleft()
            handle_mouse(ev, x, y, cw, ch)

        window.apply()
        cv2.imshow(WINDOW_NAME, display)

        if handle_key(cv2.waitKey(1) & 0xFF) or window.closed():
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Program exited.")


if __name__ == "__main__":
    main()