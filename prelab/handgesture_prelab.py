import cv2
import mediapipe as mp
import numpy as np
import math
import time
import subprocess
import sys
import ctypes
from collections import deque

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

if not (IS_WINDOWS or IS_LINUX):
    print(f"Unsupported platform: {sys.platform}. This script supports Windows and Linux.")
    sys.exit(1)

PYCAW_AVAILABLE = False
if IS_WINDOWS:
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        PYCAW_AVAILABLE = True
    except Exception:
        PYCAW_AVAILABLE = False

CAMERA_INDEX = 0
CONFIRMATION_TIME = 0.5
COOLDOWN_TIME = 0.7
SWIPE_MIN_DISTANCE = 60
SWIPE_MIN_HORIZONTAL_RATIO = 1.2
SWIPE_MAX_DURATION = 0.6
SWIPE_MIN_VELOCITY = 80.0   
MIN_DISTANCE = 0.02
MAX_DISTANCE = 0.3
OPEN_PALM_TOLERANCE = 0.02 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
class LinuxMediaController:
    """Handles Linux media control via playerctl, pactl, or wpctl."""

    def __init__(self):
        self.use_playerctl = self._check_playerctl()
        self.use_pactl = self._check_pactl()
        self.use_wpctl = self._check_wpctl()
        self.muted = False

        if not (self.use_playerctl or self.use_pactl or self.use_wpctl):
            print("WARNING: No Linux media control found. Install playerctl or ensure pactl/wpctl available.")

    def _check_playerctl(self):
        return subprocess.run(["which", "playerctl"], capture_output=True).returncode == 0

    def _check_pactl(self):
        return subprocess.run(["which", "pactl"], capture_output=True).returncode == 0

    def _check_wpctl(self):
        return subprocess.run(["which", "wpctl"], capture_output=True).returncode == 0

    def get_active_player(self):
        """Find a usable MPRIS player, preferring one that is currently playing."""
        if not self.use_playerctl:
            return None

        try:
            result = subprocess.run(
                ["playerctl", "-l"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                print(f"ERROR: Cannot list media players: {result.stderr.strip()}")
                return None

            players = [
                player.strip()
                for player in result.stdout.splitlines()
                if player.strip()
            ]

            if not players:
                print("ERROR: No MPRIS players found.")
                return None

            candidates = []

            for player in players:
                status_result = subprocess.run(
                    ["playerctl", "--player", player, "status"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if status_result.returncode == 0:
                    status = status_result.stdout.strip()
                    if status in ("Playing", "Paused", "Stopped"):
                        candidates.append((player, status))

            for player, status in candidates:
                if status == "Playing":
                    return player

            for player, status in candidates:
                if status == "Paused":
                    return player

            for player, status in candidates:
                if status == "Stopped":
                    return player

            print("ERROR: No controllable MPRIS players found.")
            return None

        except (subprocess.SubprocessError, OSError) as exc:
            print(f"ERROR: Media player discovery failed: {exc}")
            return None

    def _run_playerctl(self, command, action_name):
        """Run a playerctl command against a discovered MPRIS player."""
        if not self.use_playerctl:
            print("ERROR: playerctl is not available.")
            return False

        player = self.get_active_player()
        if not player:
            print(f"ERROR: {action_name} failed: no controllable media player.")
            return False

        try:
            result = subprocess.run(
                ["playerctl", "--player", player, command],
                capture_output=True,
                text=True,
                timeout=2
            )
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"ERROR: {action_name} failed on {player}: {exc}")
            return False

        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "unknown error"
            print(f"ERROR: {action_name} failed on {player}: {error}")
            return False

        print(f"MEDIA: {action_name} executed on {player}")
        return True

    def play_pause(self):
        return self._run_playerctl("play-pause", "Play/Pause")

    def next_track(self):
        return self._run_playerctl("next", "Next Track")

    def previous_track(self):
        return self._run_playerctl("previous", "Previous Track")

    def mute(self):
        if self.use_pactl:
            result = subprocess.run(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                print("System mute toggled")
                return True

            print(f"pactl mute failed: {result.stderr.strip()}")

        if self.use_wpctl:
            result = subprocess.run(
                ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                print("System mute toggled")
                return True

            print(f"wpctl mute failed: {result.stderr.strip()}")

        print("No system mute controller available.")
        return False

    def set_volume(self, volume):
        volume = max(0, min(100, int(volume)))

        if self.use_pactl:
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                print(f"Volume set to {volume}%")
                return True

            print(f"pactl volume failed: {result.stderr.strip()}")

        if self.use_wpctl:
            result = subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{volume}%"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                print(f"Volume set to {volume}%")
                return True

            print(f"wpctl volume failed: {result.stderr.strip()}")

        print("No system volume controller available.")
        return False

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


class WindowsMediaController:
    """Handles Windows media control via media keys and pycaw."""

    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.endpoint = None
        self.muted = False
        self._last_volume = 50  

        if PYCAW_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                if hasattr(devices, "EndpointVolume"):
                    self.endpoint = devices.EndpointVolume
                else:
                    interface = devices.Activate(
                        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    self.endpoint = ctypes.cast(
                        interface, ctypes.POINTER(IAudioEndpointVolume))
            except Exception as exc:
                print(f"WARNING: pycaw audio init failed ({exc}). Using volume keys.")
                self.endpoint = None
        else:
            print("WARNING: pycaw not installed (pip install pycaw comtypes). "
                  "Volume/mute will use media keys (approximate).")

    def _press_key(self, vk_code):
        try:
            self.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
            self.user32.keybd_event(vk_code, 0,
                                    KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            return True
        except Exception as exc:
            print(f"ERROR: key press failed: {exc}")
            return False

    def play_pause(self):
        ok = self._press_key(VK_MEDIA_PLAY_PAUSE)
        if ok:
            print("MEDIA: Play/Pause key sent")
        return ok

    def next_track(self):
        ok = self._press_key(VK_MEDIA_NEXT_TRACK)
        if ok:
            print("MEDIA: Next Track key sent")
        return ok

    def previous_track(self):
        ok = self._press_key(VK_MEDIA_PREV_TRACK)
        if ok:
            print("MEDIA: Previous Track key sent")
        return ok

    def mute(self):
        if self.endpoint is not None:
            try:
                current = bool(self.endpoint.GetMute())
                self.endpoint.SetMute(0 if current else 1, None)
                self.muted = not current
                print(f"System {'muted' if self.muted else 'unmuted'}")
                return True
            except Exception as exc:
                print(f"pycaw mute failed: {exc}")

        if self._press_key(VK_VOLUME_MUTE):
            self.muted = not self.muted
            print("System mute toggled (key)")
            return True

        print("No system mute controller available.")
        return False

    def set_volume(self, volume):
        volume = max(0, min(100, int(volume)))

        if self.endpoint is not None:
            try:
                self.endpoint.SetMasterVolumeLevelScalar(volume / 100.0, None)
                self._last_volume = volume
                print(f"Volume set to {volume}%")
                return True
            except Exception as exc:
                print(f"pycaw volume failed: {exc}")
        diff = volume - self._last_volume
        presses = int(round(abs(diff) / 2.0))
        if presses > 0:
            key = VK_VOLUME_UP if diff > 0 else VK_VOLUME_DOWN
            for _ in range(presses):
                self._press_key(key)
            self._last_volume = volume
            print(f"Volume ~{volume}% (key fallback)")
            return True

        return False


def create_media_controller():
    """Pick the media controller for the current OS."""
    if IS_WINDOWS:
        print("Platform: Windows")
        return WindowsMediaController()
    print("Platform: Linux")
    return LinuxMediaController()


media = create_media_controller()

system_locked = False
volume = 50
action_history = deque(maxlen=30)
last_action_time = 0.0
swipe_history = deque(maxlen=20)
confirm_start_time = None
confirm_pending = None
previous_static_gesture = "none"  
gesture_latched = False          


def calc_distance(p1, p2):
    """Euclidean distance between two landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_open_palm(landmarks):
    """Detect an open palm while tolerating small landmark jitter.

    A fingertip must be above/forward of its PIP joint within the configured
    tolerance, and the fingertip must also be meaningfully displaced from the
    wrist so a neutral/empty landmark set is not classified as an open palm.
    """
    wrist_y = landmarks[0].y

    extensions = [
        landmarks[8].y <= landmarks[6].y + OPEN_PALM_TOLERANCE
        and landmarks[8].y < wrist_y - OPEN_PALM_TOLERANCE / 2,
        landmarks[12].y <= landmarks[10].y + OPEN_PALM_TOLERANCE
        and landmarks[12].y < wrist_y - OPEN_PALM_TOLERANCE / 2,
        landmarks[16].y <= landmarks[14].y + OPEN_PALM_TOLERANCE
        and landmarks[16].y < wrist_y - OPEN_PALM_TOLERANCE / 2,
        landmarks[20].y <= landmarks[18].y + OPEN_PALM_TOLERANCE
        and landmarks[20].y < wrist_y - OPEN_PALM_TOLERANCE / 2,
    ]
    return all(extensions)


def is_closed_fist(landmarks):
    """Detect closed fist: all fingertips below their PIP joints."""
    flexed = [
        landmarks[8].y > landmarks[6].y,
        landmarks[12].y > landmarks[10].y,
        landmarks[16].y > landmarks[14].y,
        landmarks[20].y > landmarks[18].y,
    ]
    return all(flexed)


def is_two_finger(landmarks):
    """Detect two-finger gesture: index and middle extended, others flexed.

    Uses distance-from-wrist for rotation and perspective tolerance.
    """
    wrist = landmarks[0]

    def is_extended(tip_idx, pip_idx):
        tip_dist = calc_distance(landmarks[tip_idx], wrist)
        pip_dist = calc_distance(landmarks[pip_idx], wrist)
        return tip_dist > pip_dist + 0.02

    index_up = is_extended(8, 6)
    middle_up = is_extended(12, 10)
    ring_down = not is_extended(16, 14)
    pinky_down = not is_extended(20, 18)
    return index_up and middle_up and ring_down and pinky_down


def detect_swipe(history):
    """Detect left or right swipe from time-based motion history.

    History entries are (timestamp, x, y) tuples. Requires minimum
    horizontal displacement, horizontal dominance, and minimum velocity
    so that slow diagonal or vertical movements do not trigger a swipe.
    """
    if len(history) < 2:
        return None
    start_t, start_x, start_y = history[0]
    end_t, end_x, end_y = history[-1]
    duration = end_t - start_t
    if duration <= 0:
        return None
    dx = end_x - start_x
    dy = end_y - start_y
    if abs(dx) < SWIPE_MIN_DISTANCE:
        return None
    if abs(dx) < abs(dy) * SWIPE_MIN_HORIZONTAL_RATIO:
        return None
    if duration > SWIPE_MAX_DURATION:
        return None
    if abs(dx) / duration < SWIPE_MIN_VELOCITY:
        return None
    return "right" if dx > 0 else "left"


def in_cooldown():
    """Check if we are currently in cooldown period."""
    return time.monotonic() - last_action_time < COOLDOWN_TIME


def trigger_action(action):
    """Record an action with timestamp."""
    global last_action_time
    last_action_time = time.monotonic()
    ts = time.strftime("%H:%M:%S")
    action_history.append((ts, action))
    print(f"[{ts}] {action}")


def recognize_gesture(landmarks, w, h):
    """Recognise a gesture. Returns (gesture_type, description, volume).

    Recognition only - action execution is handled centrally in main().
    Volume is calculated but NOT applied here so the locked state can be
    enforced consistently at the execution layer.
    """
    global volume
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = calc_distance(thumb_tip, index_tip)

    normalized_dist = max(0.0, min(1.0, (dist - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)))
    new_volume = int(normalized_dist * 100)

    if is_two_finger(landmarks):
        return "two_finger", "Two Fingers", None

    swipe = detect_swipe(swipe_history)
    if swipe == "right":
        swipe_history.clear()
        return "swipe_right", "Right Swipe", None
    if swipe == "left":
        swipe_history.clear()
        return "swipe_left", "Left Swipe", None

    if is_open_palm(landmarks):
        return "open_palm", "Open Palm", None

    if is_closed_fist(landmarks):
        return "fist", "Closed Fist", None

    if abs(new_volume - volume) > 3:
        return "volume", f"Thumb-Index: {dist:.3f}, Vol: {new_volume}%", new_volume

    return "none", "Idle", None


def render_dashboard(frame, gesture_type, gesture_desc, fps,
                     status_text, confirm_label, cooldown_remaining):
    """Draw the info dashboard on the frame."""
    global system_locked
    h, w = frame.shape[:2]
    dash_y = h - 170

    lock_color = (255, 0, 0) if system_locked else (0, 255, 0)
    detected_color = (0, 255, 0) if gesture_type != "none" else (150, 150, 150)

    texts = [
        ("SOUTHVILLE INTERNATIONAL SCHOOL AND COLLEGES", (255, 255, 255), 0),
        ("ADVANCED OPENCV LABORATORY", (255, 255, 255), 22),
        ("", (0, 0, 0), 44),
        (f"SYSTEM: {'LOCKED' if system_locked else 'UNLOCKED'}", lock_color, 44),
        (f"GESTURE: {gesture_desc.upper()}", detected_color, 68),
        (f"STATUS: {status_text}", (0, 200, 255), 90),
        (f"VOLUME: {volume}%", (255, 200, 0), 112),
        (f"MOTION: {len(swipe_history)} pts", (180, 180, 255), 134),
    ]

    if system_locked:
        texts.append(("LOCKED - COMMANDS DISABLED", (255, 80, 80), 156))
    if confirm_label:
        texts.append((f"CONFIRMING: {confirm_label}", (0, 255, 255), 178))
    if cooldown_remaining > 0:
        texts.append((f"COOLDOWN: {cooldown_remaining:.1f}s", (255, 120, 120), 198))

    if len(swipe_history) >= 2:
        dx = swipe_history[-1][1] - swipe_history[0][1]
        dy = swipe_history[-1][2] - swipe_history[0][2]
        texts.append((f"SWIPE DX: {dx:.0f}  DY: {dy:.0f}", (255, 200, 0), 218))

    for text, color, y_offset in texts:
        cv2.putText(frame, text, (20, dash_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, dash_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def camera_candidates():
    """Return (source, backend, label) camera candidates for this OS."""
    if IS_WINDOWS:
        return [
            (CAMERA_INDEX, cv2.CAP_DSHOW, f"index {CAMERA_INDEX} (DSHOW)"),
            (CAMERA_INDEX, cv2.CAP_MSMF, f"index {CAMERA_INDEX} (MSMF)"),
            (CAMERA_INDEX, cv2.CAP_ANY, f"index {CAMERA_INDEX} (ANY)"),
        ]
    return [
        ("/dev/video0", None, "/dev/video0"),
        (CAMERA_INDEX, cv2.CAP_V4L2, f"index {CAMERA_INDEX} (V4L2)"),
        (CAMERA_INDEX, cv2.CAP_ANY, f"index {CAMERA_INDEX} (ANY)"),
    ]


def initialize_camera():
    """Initialize camera with a robust, OS-specific fallback strategy.

    Windows: DirectShow -> Media Foundation -> any backend.
    Linux:   /dev/video0 -> V4L2 index -> any backend.
    """
    candidates = camera_candidates()

    for source, backend, desc in candidates:
        if backend is None:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, backend)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera device: {desc}")
                print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
                return cap
            else:
                cap.release()
                print(f"Warning: {desc} opened but first frame failed")
        else:
            cap.release()

    hint = ("Check Settings > Privacy & security > Camera and close other apps using the camera."
            if IS_WINDOWS else
            "Check that /dev/video0 exists and your user is in the 'video' group.")
    raise RuntimeError(f"Unable to open webcam. Tried: {[c[2] for c in candidates]}. {hint}")


def main():
    global system_locked, confirm_start_time, confirm_pending
    global previous_static_gesture, gesture_latched

    cap = None
    try:
        cap = initialize_camera()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    consecutive_failures = 0
    max_failures_before_reopen = 30

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print("Warning: webcam frame read failed (attempting recovery).")
            elif consecutive_failures >= max_failures_before_reopen:
                print(f"ERROR: Webcam frame acquisition failed {consecutive_failures} times.")
                break
            continue
        consecutive_failures = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = now

        now_mono = time.monotonic()
        results = hands.process(rgb)
        gesture_type = "none"
        gesture_desc = "No Hand"
        status_text = "WAITING"
        confirm_label = None
        cooldown_remaining = 0.0

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0],
                                   mp_hands.HAND_CONNECTIONS)

            wrist = landmarks[0]
            wrist_x = w - wrist.x * w
            wrist_y = wrist.y * h
            while swipe_history and now_mono - swipe_history[0][0] > SWIPE_MAX_DURATION:
                swipe_history.popleft()
            swipe_history.append((now_mono, wrist_x, wrist_y))

            gesture_type, gesture_desc, new_volume = recognize_gesture(landmarks, w, h)
            if gesture_type == "two_finger":
                if not gesture_latched:
                    if confirm_pending != ("unlock" if system_locked else "lock"):
                        confirm_pending = "unlock" if system_locked else "lock"
                        confirm_start_time = now_mono
                        status_text = "CONFIRMING"
                        confirm_label = confirm_pending.upper()
                    elif now_mono - confirm_start_time >= CONFIRMATION_TIME:
                        if confirm_pending == "lock":
                            system_locked = True
                            trigger_action("System LOCKED")
                        else:
                            system_locked = False
                            trigger_action("System UNLOCKED")
                        confirm_pending = None
                        confirm_start_time = None
                        gesture_latched = True
                        swipe_history.clear()
                if confirm_pending:
                    status_text = "CONFIRMING"
                    confirm_label = confirm_pending.upper()
            elif system_locked:
                if confirm_pending:
                    confirm_pending = None
                    confirm_start_time = None
                gesture_type = "none"
                gesture_desc = "LOCKED"
                status_text = "LOCKED"
                gesture_latched = False
            else:
                if gesture_type == "open_palm":
                    if previous_static_gesture != "open_palm":
                        if not in_cooldown():
                            confirm_pending = "play_pause"
                            confirm_start_time = now_mono
                            status_text = "CONFIRMING"
                            confirm_label = "PLAY/PAUSE"
                    if confirm_pending == "play_pause" and now_mono - confirm_start_time >= CONFIRMATION_TIME:
                        if media.play_pause():
                            trigger_action("Play/Pause toggled")
                            status_text = "DETECTED"
                        else:
                            trigger_action("Play/Pause failed")
                            status_text = "MEDIA ERROR"
                        confirm_pending = None
                        confirm_start_time = None
                        gesture_latched = True
                elif gesture_type in ("swipe_right", "swipe_left"):
                    if not in_cooldown():
                        direction = "next" if gesture_type == "swipe_right" else "prev"
                        success = (
                            media.next_track()
                            if direction == "next"
                            else media.previous_track()
                        )

                        if success:
                            trigger_action(f"Track {direction}")
                            status_text = "DETECTED"
                        else:
                            trigger_action(f"Track {direction} failed")
                            status_text = "MEDIA ERROR"

                        swipe_history.clear()
                        gesture_latched = True
                elif gesture_type == "fist":
                    if not in_cooldown():
                        media.mute()
                        trigger_action("Mute toggled")
                        gesture_latched = True
                        status_text = "DETECTED"
                elif gesture_type == "volume" and new_volume is not None:
                    if not in_cooldown():
                        volume = new_volume
                        media.set_volume(volume)
                        trigger_action(f"Volume -> {volume}%")
                        status_text = "DETECTED"
                if gesture_type != previous_static_gesture:
                    gesture_latched = False
                if gesture_type in ("open_palm", "fist", "two_finger", "volume"):
                    if gesture_type != previous_static_gesture:
                        if gesture_type == "open_palm":
                            print(f"Gesture transition: {previous_static_gesture.upper()} -> OPEN_PALM")
                        elif gesture_type == "fist":
                            print(f"Gesture transition: {previous_static_gesture.upper()} -> FIST")
                        elif gesture_type == "volume":
                            print(f"Gesture transition: {previous_static_gesture.upper()} -> VOLUME")
                    previous_static_gesture = gesture_type
                else:
                    if previous_static_gesture != "none":
                        print(f"Gesture transition: {previous_static_gesture.upper()} -> NONE")
                    previous_static_gesture = "none"

        else:
            if swipe_history:
                swipe_history.clear()
            if gesture_latched:
                gesture_latched = False
            if previous_static_gesture != "none":
                print(f"Gesture transition: {previous_static_gesture.upper()} -> NONE (hand lost)")
            previous_static_gesture = "none"
            if confirm_pending:
                confirm_pending = None
                confirm_start_time = None

        if confirm_pending and confirm_start_time is not None:
            elapsed = now_mono - confirm_start_time if results.multi_hand_landmarks else 0
            bar_w = int(min(elapsed / CONFIRMATION_TIME, 1.0) * 200)
            bx, by = w // 2 - 100, 60
            cv2.rectangle(frame, (bx, by), (bx + 200, by + 12), (40, 40, 60), -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 12),
                          (0, 255, 0) if elapsed < CONFIRMATION_TIME else (255, 0, 0), -1)
            label = "UNLOCK" if confirm_pending == "unlock" else \
                    "LOCK" if confirm_pending == "lock" else "PLAY/PAUSE"
            cv2.putText(frame, f"Confirming: {label}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if last_action_time > 0 and in_cooldown():
            cooldown_remaining = COOLDOWN_TIME - (time.monotonic() - last_action_time)
            if cooldown_remaining < 0:
                cooldown_remaining = 0.0

        for i, (ts, act) in enumerate(list(action_history)[-6:]):
            cv2.putText(frame, f"[{ts}] {act}", (w - 300, 140 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

        render_dashboard(frame, gesture_type, gesture_desc, fps_display,
                         status_text, confirm_label, cooldown_remaining)

        cv2.imshow("HandGesture - PreLab", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()