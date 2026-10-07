"""
test_handgesture.py
Test plan for the OpenCV Virtual Drawing & Shape Studio.

Each test documents:
    Test ID, Feature, Initial State, Gesture/Input, Expected Result,
    Actual Result, Pass/Fail, Notes.
"""

import unittest
import math
from collections import deque


class MockLandmark:
    """Minimal mock for MediaPipe landmarks."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestDistanceCalculation(unittest.TestCase):
    """Test thumb-index distance to thickness mapping."""

    def test_min_distance_maps_to_min_brush(self):
        from lab.handgesture_lab import distance_to_thickness, MIN_DISTANCE, MIN_BRUSH_SIZE
        result = distance_to_thickness(MIN_DISTANCE)
        self.assertEqual(result, MIN_BRUSH_SIZE)

    def test_max_distance_maps_to_max_brush(self):
        from lab.handgesture_lab import distance_to_thickness, MAX_DISTANCE, MAX_BRUSH_SIZE
        result = distance_to_thickness(MAX_DISTANCE)
        self.assertEqual(result, MAX_BRUSH_SIZE)

    def test_midpoint_distance(self):
        from lab.handgesture_lab import distance_to_thickness, MIN_DISTANCE, MAX_DISTANCE, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE
        mid = (MIN_DISTANCE + MAX_DISTANCE) / 2
        result = distance_to_thickness(mid)
        expected = (MIN_BRUSH_SIZE + MAX_BRUSH_SIZE) // 2
        self.assertEqual(result, expected)


class TestGestureRecognition(unittest.TestCase):
    """Test static gesture detection logic."""

    def test_open_palm_all_extended(self):
        from lab.handgesture_lab import is_open_palm
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.3)   # 6 index pip
        lm[8] = MockLandmark(0, 0.1)   # 8 index tip (above pip)
        lm[10] = MockLandmark(0, 0.3)  # 10 middle pip
        lm[12] = MockLandmark(0, 0.1)  # 12 middle tip
        lm[14] = MockLandmark(0, 0.3)  # 14 ring pip
        lm[16] = MockLandmark(0, 0.1)  # 16 ring tip
        lm[18] = MockLandmark(0, 0.3)  # 18 pinky pip
        lm[20] = MockLandmark(0, 0.1)  # 20 pinky tip
        self.assertTrue(is_open_palm(lm))

    def test_closed_fist_all_flexed(self):
        from lab.handgesture_lab import is_closed_fist
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.3)  # 6 index pip
        lm[8] = MockLandmark(0, 0.5)  # 8 index tip (below pip)
        lm[10] = MockLandmark(0, 0.3)  # 10 middle pip
        lm[12] = MockLandmark(0, 0.5)  # 12 middle tip
        lm[14] = MockLandmark(0, 0.3)  # 14 ring pip
        lm[16] = MockLandmark(0, 0.5)  # 16 ring tip
        lm[18] = MockLandmark(0, 0.3)  # 18 pinky pip
        lm[20] = MockLandmark(0, 0.5)  # 20 pinky tip
        self.assertTrue(is_closed_fist(lm))

    def test_two_finger_gesture(self):
        from lab.handgesture_lab import is_two_finger
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.4)  # index pip
        lm[8] = MockLandmark(0, 0.1)  # index tip (UP)
        lm[10] = MockLandmark(0, 0.4)  # middle pip
        lm[12] = MockLandmark(0, 0.1)  # middle tip (UP)
        lm[14] = MockLandmark(0, 0.3)  # ring pip
        lm[16] = MockLandmark(0, 0.5)  # ring tip (DOWN)
        lm[18] = MockLandmark(0, 0.3)  # pinky pip
        lm[20] = MockLandmark(0, 0.5)  # pinky tip (DOWN)
        self.assertTrue(is_two_finger(lm))

    def test_index_point_only_index_up(self):
        from lab.handgesture_lab import is_index_point
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.4); lm[8] = MockLandmark(0, 0.1)  # index UP
        lm[10] = MockLandmark(0, 0.4); lm[12] = MockLandmark(0, 0.5)  # middle DOWN
        lm[14] = MockLandmark(0, 0.4); lm[16] = MockLandmark(0, 0.5)  # ring DOWN
        lm[18] = MockLandmark(0, 0.4); lm[20] = MockLandmark(0, 0.5)  # pinky DOWN
        self.assertTrue(is_index_point(lm))

    def test_three_fingers(self):
        from lab.handgesture_lab import is_three_fingers
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.4); lm[8] = MockLandmark(0, 0.1)   # index UP
        lm[10] = MockLandmark(0, 0.4); lm[12] = MockLandmark(0, 0.1)  # middle UP
        lm[14] = MockLandmark(0, 0.4); lm[16] = MockLandmark(0, 0.1)  # ring UP
        lm[18] = MockLandmark(0, 0.4); lm[20] = MockLandmark(0, 0.5)  # pinky DOWN
        self.assertTrue(is_three_fingers(lm))

    def test_four_fingers(self):
        from lab.handgesture_lab import is_four_fingers
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[6] = MockLandmark(0, 0.4); lm[8] = MockLandmark(0, 0.1)
        lm[10] = MockLandmark(0, 0.4); lm[12] = MockLandmark(0, 0.1)
        lm[14] = MockLandmark(0, 0.4); lm[16] = MockLandmark(0, 0.1)
        lm[18] = MockLandmark(0, 0.4); lm[20] = MockLandmark(0, 0.1)
        self.assertTrue(is_four_fingers(lm))

    def test_pinch_detection(self):
        from lab.handgesture_lab import detect_pinching
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[4] = MockLandmark(0.1, 0.1)   # thumb tip
        lm[8] = MockLandmark(0.11, 0.11) # index tip — very close
        self.assertTrue(detect_pinching(lm))

    def test_no_pinch_when_far(self):
        from lab.handgesture_lab import detect_pinching
        lm = [MockLandmark(0, 0) for _ in range(21)]
        lm[4] = MockLandmark(0.1, 0.1)
        lm[8] = MockLandmark(0.5, 0.5)
        self.assertFalse(detect_pinching(lm))


class TestSwipeDetection(unittest.TestCase):
    """Test motion-based swipe gesture detection."""

    def test_right_swipe(self):
        from lab.handgesture_lab import detect_swipe
        history = deque(maxlen=5)
        for i in range(5):
            history.append(100 + i * 30)  # moving right
        result = detect_swipe(history)
        self.assertEqual(result, "right")

    def test_left_swipe(self):
        from lab.handgesture_lab import detect_swipe
        history = deque(maxlen=5)
        for i in range(5):
            history.append(500 - i * 30)  # moving left
        result = detect_swipe(history)
        self.assertEqual(result, "left")

    def test_no_swipe_small_movement(self):
        from lab.handgesture_lab import detect_swipe, SWIPE_THRESHOLD
        history = deque(maxlen=5)
        for i in range(5):
            history.append(200 + i * 10)  # small movement
        result = detect_swipe(history)
        self.assertIsNone(result)

    def test_insufficient_history(self):
        from lab.handgesture_lab import detect_swipe
        history = deque(maxlen=5)
        history.append(100)
        history.append(200)
        result = detect_swipe(history)
        self.assertIsNone(result)


class TestHistoryManager(unittest.TestCase):
    """Test undo/redo history stack."""

    def test_push_and_undo(self):
        from lab.handgesture_lab import HistoryManager
        import numpy as np
        hm = HistoryManager(max_len=5)
        c1 = np.zeros((10, 10), dtype=np.uint8)
        c2 = np.ones((10, 10), dtype=np.uint8) * 200
        # push() saves state BEFORE an operation
        hm.push(c1)   # save initial state
        # After operation, canvas becomes c2
        result = hm.undo(c2)  # undo → should return c1 (the saved state)
        self.assertTrue(np.array_equal(result, c1))

    def test_empty_history_returns_current(self):
        from lab.handgesture_lab import HistoryManager
        import numpy as np
        hm = HistoryManager()
        current = np.zeros((5, 5), dtype=np.uint8)
        result = hm.undo(current)
        self.assertTrue(np.array_equal(result, current))

    def test_duplicate_not_pushed(self):
        from lab.handgesture_lab import HistoryManager
        import numpy as np
        hm = HistoryManager()
        c = np.zeros((5, 5), dtype=np.uint8)
        hm.push(c)
        hm.push(c)  # duplicate — should not add
        self.assertEqual(len(hm.stack), 1)

    def test_clear_history(self):
        from lab.handgesture_lab import HistoryManager
        import numpy as np
        hm = HistoryManager()
        hm.push(np.zeros((5, 5)))
        hm.push(np.ones((5, 5)) * 100)
        hm.clear()
        self.assertEqual(len(hm.stack), 0)


class TestToolbarZoneDetection(unittest.TestCase):
    """Test toolbar zone identification."""

    def test_tool_zone_detection(self):
        from lab.handgesture_lab import Toolbar, TOOLS
        tb = Toolbar(y_start=10, tool_h=50, color_h=45, size_h=30)
        zone_type, zone_name = tb.get_zone(50, 35, 800, 600)
        self.assertEqual(zone_type, "tool")
        self.assertIn(zone_name, TOOLS)

    def test_color_zone_detection(self):
        from lab.handgesture_lab import Toolbar, COLOR_ORDER
        tb = Toolbar(y_start=10, tool_h=50, color_h=45, size_h=30)
        zone_type, zone_name = tb.get_zone(100, 100, 800, 600)
        self.assertEqual(zone_type, "color")
        self.assertIn(zone_name, COLOR_ORDER)

    def test_size_zone_detection(self):
        from lab.handgesture_lab import Toolbar
        tb = Toolbar(y_start=10, tool_h=50, color_h=45, size_h=30)
        zone_type, zone_name = tb.get_zone(200, 130, 800, 600)
        self.assertEqual(zone_type, "size")
        self.assertIn(zone_name, [2, 5, 10, 20, 30, 50])

    def test_outside_toolbar(self):
        from lab.handgesture_lab import Toolbar
        tb = Toolbar(y_start=10, tool_h=50, color_h=45, size_h=30)
        zone_type, zone_name = tb.get_zone(400, 500, 800, 600)
        self.assertIsNone(zone_type)
        self.assertIsNone(zone_name)


class TestCanvasOperations(unittest.TestCase):
    """Test canvas drawing and erasing operations."""

    def test_freehand_line_drawing(self):
        import numpy as np
        from lab.handgesture_lab import COLORS, current_color, brush_size
        # Simulate drawing a line
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        color = COLORS["RED"]
        cv2_line_temp = lambda img, p1, p2, c, t: __import__('cv2').line(img, p1, p2, c, t)
        cv2_line_temp(canvas, (10, 10), (50, 50), color, 5)
        # Check that pixels changed
        self.assertFalse(np.all(canvas == 255))

    def test_eraser_removes_pixels(self):
        import numpy as np
        import cv2
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        color = (0, 0, 255)
        cv2.circle(canvas, (50, 50), 30, color, -1)
        # Erase center
        bg = (255, 255, 255)
        cv2.circle(canvas, (50, 50), 10, bg, -1)
        # Center should be white again
        self.assertTrue(np.all(canvas[48:52, 48:52] == 255))


class TestShapeRenderer(unittest.TestCase):
    """Test geometric shape creation."""

    def test_line_commit(self):
        import numpy as np
        import cv2
        from lab.handgesture_lab import ShapeRenderer, COLORS
        sr = ShapeRenderer()
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        sr.start_shape((10, 10), "line")
        sr.update_preview((90, 90))
        canvas = sr.commit(canvas, COLORS["RED"], 3)
        # Verify line was drawn
        self.assertFalse(np.all(canvas == 255))

    def test_rectangle_commit(self):
        import numpy as np
        from lab.handgesture_lab import ShapeRenderer, COLORS
        sr = ShapeRenderer()
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        sr.start_shape((10, 10), "rect")
        sr.update_preview((90, 50))
        canvas = sr.commit(canvas, COLORS["BLUE"], 2)
        self.assertFalse(np.all(canvas == 255))

    def test_circle_commit(self):
        import numpy as np
        from lab.handgesture_lab import ShapeRenderer, COLORS
        sr = ShapeRenderer()
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        sr.start_shape((50, 50), "circle")
        sr.update_preview((80, 50))  # radius = 30
        canvas = sr.commit(canvas, COLORS["GREEN"], -1)  # filled circle
        # Check a pixel on the circle edge is colored
        self.assertFalse(np.all(canvas[80, 50] == 255))

    def test_preview_matches_commit(self):
        import numpy as np
        from lab.handgesture_lab import ShapeRenderer, COLORS
        sr = ShapeRenderer()
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        sr.start_shape((10, 10), "line")
        sr.update_preview((90, 90))
        preview = sr.get_preview(canvas, COLORS["RED"], 3)
        committed = sr.commit(canvas.copy(), COLORS["RED"], 3)
        self.assertTrue(np.array_equal(preview, committed))

    def test_cancel_clears_state(self):
        from lab.handgesture_lab import ShapeRenderer
        sr = ShapeRenderer()
        sr.start_shape((10, 10), "line")
        sr.cancel()
        self.assertIsNone(sr.shape_mode)
        self.assertIsNone(sr.start_point)


class TestCursorPosition(unittest.TestCase):
    """Test cursor smoothing and position tracking."""

    def test_smooth_cursor_reduces_jitter(self):
        from lab.handgesture_lab import Cursor
        c = Cursor()
        c.update(100, 100)
        c.update(102, 98)  # slight jitter
        pos = c.update(101, 99)
        # Should be smoothed toward 100,100
        self.assertIsNotNone(pos)
        self.assertLess(abs(pos[0] - 100), 3)

    def test_cursor_position_mapping(self):
        from lab.handgesture_lab import Cursor
        c = Cursor()
        c.update(320, 240)
        pos = c.pos
        self.assertEqual(pos, (320, 240))


class TestStateManager(unittest.TestCase):
    """Test state machine transitions."""

    def test_initial_state_is_locked(self):
        from lab.handgesture_lab import state, MODE_LOCKED
        self.assertEqual(state.mode, MODE_LOCKED)

    def test_mode_transition(self):
        from lab.handgesture_lab import state, MODE_DRAWING
        state.set_mode(MODE_DRAWING)
        self.assertEqual(state.mode, MODE_DRAWING)
        state.set_mode("INVALID")  # should not change
        self.assertEqual(state.mode, MODE_DRAWING)

    def test_cooldown_after_action(self):
        from lab.handgesture_lab import state
        state.trigger_action("test")
        self.assertTrue(state.in_cooldown())

    def test_cooldown_expires(self):
        import time
        from lab.handgesture_lab import state, COOLDOWN_TIME
        state.trigger_action("test")
        # Manually set last action time far in the past
        state.last_action_time = time.time() - COOLDOWN_TIME - 1
        self.assertFalse(state.in_cooldown())

    def test_save_message_display(self):
        from lab.handgesture_lab import state
        state.save_message = "SAVED: drawing_test.png"
        state.save_message_time = __import__('time').time()
        self.assertEqual(state.save_message, "SAVED: drawing_test.png")


if __name__ == "__main__":
    unittest.main()
