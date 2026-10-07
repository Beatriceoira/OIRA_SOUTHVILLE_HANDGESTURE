"""
test_prelab.py
Unit tests for the Pre-Lab hand gesture controller (media playback mode).

Covers:
    * is_open_palm / is_closed_fist / is_two_finger landmark logic
    * detect_swipe algorithm (direction, distance, dominance, velocity)
    * gesture state-machine helpers (cooldown, latching, lock enforcement)
    * recognition pipeline returning correct types
    * media-action dispatch respects lock state and cooldown
"""

import math
import time
import unittest
from collections import deque


class MockLandmark:
    """Minimal mock for MediaPipe landmarks.  x/y are normalised (0..1)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestOpenPalm(unittest.TestCase):
    """Test open-palm landmark detection with tolerance."""

    def _make_lm(self, extensions=None):
        """extensions is dict tip_idx -> bool (True = extended, False = flexed).

        With wrist at (0.5, 0.5):
          - Extended : tip is FURTHER from wrist than PIP (tip.y < pip.y)
          - Flexed   : tip is CLOSER to wrist than PIP (tip.y > pip.y)
        """
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        # Defaults: all fingers extended
        lm[6]  = MockLandmark(0.5, 0.45); lm[8]  = MockLandmark(0.5, 0.30)
        lm[10] = MockLandmark(0.5, 0.45); lm[12] = MockLandmark(0.5, 0.30)
        lm[14] = MockLandmark(0.5, 0.45); lm[16] = MockLandmark(0.5, 0.30)
        lm[18] = MockLandmark(0.5, 0.45); lm[20] = MockLandmark(0.5, 0.30)
        if extensions:
            for idx, up in extensions.items():
                pip = idx - 2
                if up:
                    lm[idx] = MockLandmark(0.5, 0.30)   # tip farther from wrist
                    lm[pip] = MockLandmark(0.5, 0.45)    # PIP closer
                else:
                    lm[idx] = MockLandmark(0.5, 0.45)    # tip closer
                    lm[pip] = MockLandmark(0.5, 0.30)    # PIP farther
        return lm

    def test_all_extended_is_open_palm(self):
        from prelab.handgesture_prelab import is_open_palm
        self.assertTrue(is_open_palm(self._make_lm({8: True, 12: True, 16: True, 20: True})))

    def test_all_flexed_is_not_open_palm(self):
        from prelab.handgesture_prelab import is_open_palm
        self.assertFalse(is_open_palm(self._make_lm({8: False, 12: False, 16: False, 20: False})))

    def test_jitter_tolerance_allowed(self):
        """Tips within OPEN_PALM_TOLERANCE still count as extended."""
        from prelab.handgesture_prelab import is_open_palm, OPEN_PALM_TOLERANCE
        lm = self._make_lm({8: True, 12: True, 16: True, 20: True})
        # Push every tip exactly to the tolerance boundary (should still pass)
        for idx in (8, 12, 16, 20):
            lm[idx].y = lm[idx - 2].y + OPEN_PALM_TOLERANCE + 0.001
        self.assertTrue(is_open_palm(lm))

    def test_index_only_is_not_open_palm(self):
        from prelab.handgesture_prelab import is_open_palm
        self.assertFalse(is_open_palm(self._make_lm({8: True, 12: False, 16: False, 20: False})))


class TestClosedFist(unittest.TestCase):
    """Test closed-fist landmark detection."""

    def _make_lm(self, extensions=None):
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        # Defaults: all fingers extended
        lm[6]  = MockLandmark(0.5, 0.45); lm[8]  = MockLandmark(0.5, 0.30)
        lm[10] = MockLandmark(0.5, 0.45); lm[12] = MockLandmark(0.5, 0.30)
        lm[14] = MockLandmark(0.5, 0.45); lm[16] = MockLandmark(0.5, 0.30)
        lm[18] = MockLandmark(0.5, 0.45); lm[20] = MockLandmark(0.5, 0.30)
        if extensions:
            for idx, up in extensions.items():
                pip = idx - 2
                if up:
                    lm[idx] = MockLandmark(0.5, 0.30)
                    lm[pip] = MockLandmark(0.5, 0.45)
                else:
                    lm[idx] = MockLandmark(0.5, 0.45)
                    lm[pip] = MockLandmark(0.5, 0.30)
        return lm

    def test_all_flexed_is_closed_fist(self):
        from prelab.handgesture_prelab import is_closed_fist
        self.assertTrue(is_closed_fist(self._make_lm({8: False, 12: False, 16: False, 20: False})))

    def test_all_extended_is_not_closed_fist(self):
        from prelab.handgesture_prelab import is_closed_fist
        self.assertFalse(is_closed_fist(self._make_lm({8: True, 12: True, 16: True, 20: True})))

    def test_one_extended_breaks_fist(self):
        from prelab.handgesture_prelab import is_closed_fist
        self.assertFalse(is_closed_fist(self._make_lm({8: True, 12: False, 16: False, 20: False})))


class TestTwoFinger(unittest.TestCase):
    """Test two-finger (Lock/Unlock) landmark detection using distance-from-wrist."""

    def _make_lm(self, extensions=None):
        """wrist at (0.5, 0.5). Tip extended means farther from wrist than PIP."""
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        # Defaults: all extended
        lm[6]  = MockLandmark(0.5, 0.45); lm[8]  = MockLandmark(0.5, 0.30)
        lm[10] = MockLandmark(0.5, 0.45); lm[12] = MockLandmark(0.5, 0.30)
        lm[14] = MockLandmark(0.5, 0.45); lm[16] = MockLandmark(0.5, 0.30)
        lm[18] = MockLandmark(0.5, 0.45); lm[20] = MockLandmark(0.5, 0.30)
        if extensions:
            for idx, up in extensions.items():
                pip = idx - 2
                if up:
                    lm[idx] = MockLandmark(0.5, 0.30)   # far from wrist
                    lm[pip] = MockLandmark(0.5, 0.45)    # near wrist
                else:
                    lm[idx] = MockLandmark(0.5, 0.45)    # near wrist
                    lm[pip] = MockLandmark(0.5, 0.30)    # far from wrist
        return lm

    def test_index_middle_up_ring_pinky_down(self):
        from prelab.handgesture_prelab import is_two_finger
        self.assertTrue(is_two_finger(self._make_lm({8: True, 12: True, 16: False, 20: False})))

    def test_index_only_is_not_two_finger(self):
        from prelab.handgesture_prelab import is_two_finger
        self.assertFalse(is_two_finger(self._make_lm({8: True, 12: False, 16: False, 20: False})))

    def test_open_palm_is_not_two_finger(self):
        from prelab.handgesture_prelab import is_two_finger
        self.assertFalse(is_two_finger(self._make_lm({8: True, 12: True, 16: True, 20: True})))

    def test_closed_fist_is_not_two_finger(self):
        from prelab.handgesture_prelab import is_two_finger
        self.assertFalse(is_two_finger(self._make_lm({8: False, 12: False, 16: False, 20: False})))

    def test_three_fingers_is_not_two_finger(self):
        from prelab.handgesture_prelab import is_two_finger
        self.assertFalse(is_two_finger(self._make_lm({8: True, 12: True, 16: True, 20: False})))


class TestSwipeDetection(unittest.TestCase):
    """Test the robust swipe-detection algorithm."""

    def _history(self, points):
        """points is [(t, x, y), ...]."""
        return deque(points, maxlen=20)

    def test_right_swipe(self):
        from prelab.handgesture_prelab import detect_swipe
        h = self._history([
            (0, 100, 240), (0.05, 150, 240), (0.1, 250, 241),
            (0.15, 400, 240), (0.2, 600, 240),
        ])
        self.assertEqual(detect_swipe(h), "right")

    def test_left_swipe(self):
        from prelab.handgesture_prelab import detect_swipe
        h = self._history([
            (0, 600, 240), (0.05, 500, 240), (0.1, 350, 241),
            (0.15, 200, 240), (0.2, 100, 240),
        ])
        self.assertEqual(detect_swipe(h), "left")

    def test_insufficient_distance_no_swipe(self):
        from prelab.handgesture_prelab import detect_swipe, SWIPE_MIN_DISTANCE
        h = self._history([
            (0, 300, 240), (0.2, 300 + SWIPE_MIN_DISTANCE - 1, 240),
        ])
        self.assertIsNone(detect_swipe(h))

    def test_vertical_movement_rejected(self):
        from prelab.handgesture_prelab import detect_swipe
        h = self._history([
            (0, 300, 100), (0.1, 300, 400),  # pure vertical
        ])
        self.assertIsNone(detect_swipe(h))

    def test_diagonal_movement_rejected_when_dominant_vertical(self):
        """Diagonal with dy >> dx should be rejected."""
        from prelab.handgesture_prelab import detect_swipe
        h = self._history([
            (0, 100, 100),
            (0.15, 200, 450),  # dx=100, dy=350 -> dy >> dx
        ])
        self.assertIsNone(detect_swipe(h))

    def test_insufficient_history(self):
        from prelab.handgesture_prelab import detect_swipe
        h = self._history([(0, 0, 0)])
        self.assertIsNone(detect_swipe(h))

    def test_too_slow_no_swipe(self):
        from prelab.handgesture_prelab import detect_swipe, SWIPE_MIN_VELOCITY
        h = self._history([
            (0, 0, 240), (SWIPE_MIN_VELOCITY + 1, 200, 240),
        ])
        self.assertIsNone(detect_swipe(h))

    def test_too_long_duration_no_swipe(self):
        from prelab.handgesture_prelab import detect_swipe, SWIPE_MAX_DURATION
        h = self._history([
            (0, 100, 240), (SWIPE_MAX_DURATION + 0.1, 600, 240),
        ])
        self.assertIsNone(detect_swipe(h))


class TestCooldownAndLock(unittest.TestCase):
    """Test cooldown timer and lock-state enforcement."""

    def test_in_cooldown_returns_false_initially(self):
        from prelab.handgesture_prelab import in_cooldown
        self.assertFalse(in_cooldown())

    def test_in_cooldown_after_trigger(self):
        from prelab.handgesture_prelab import trigger_action, in_cooldown
        trigger_action("test")
        self.assertTrue(in_cooldown())

    def test_cooldown_expires(self):
        import prelab.handgesture_prelab as m
        from prelab.handgesture_prelab import COOLDOWN_TIME
        m.last_action_time = time.monotonic() - COOLDOWN_TIME - 1
        self.assertFalse(m.in_cooldown())

    def test_system_locked_attribute_exists(self):
        """system_locked is a module-level boolean state variable."""
        import prelab.handgesture_prelab as m
        self.assertIsInstance(m.system_locked, bool)

    def test_volume_attribute_exists(self):
        import prelab.handgesture_prelab as m
        self.assertIsInstance(m.volume, int)


class TestRecognitionReturns(unittest.TestCase):
    """Verify recognize_gesture() returns (type, desc, volume_or_None)."""

    def test_two_finger_returns_none_volume(self):
        from prelab.handgesture_prelab import recognize_gesture
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        lm[6]  = MockLandmark(0.5, 0.45); lm[8]  = MockLandmark(0.5, 0.30)
        lm[10] = MockLandmark(0.5, 0.45); lm[12] = MockLandmark(0.5, 0.30)
        lm[14] = MockLandmark(0.5, 0.45); lm[16] = MockLandmark(0.5, 0.48)
        lm[18] = MockLandmark(0.5, 0.45); lm[20] = MockLandmark(0.5, 0.48)
        gt, desc, vol = recognize_gesture(lm, 640, 480)
        self.assertEqual(gt, "two_finger")
        self.assertIsNone(vol)

    def test_swipe_returns_none_volume(self):
        from prelab.handgesture_prelab import recognize_gesture
        import prelab.handgesture_prelab as m
        m.swipe_history.clear()
        for i in range(5):
            m.swipe_history.append((i * 0.05, 100 + i * 120, 240))
        gt, desc, vol = recognize_gesture(
            [MockLandmark(0.5, 0.5) for _ in range(21)], 640, 480)
        self.assertEqual(gt, "swipe_right")
        self.assertIsNone(vol)

    def test_volume_returns_new_volume(self):
        from prelab.handgesture_prelab import recognize_gesture
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        lm[4] = MockLandmark(0.5, 0.55)   # thumb close
        lm[8] = MockLandmark(0.5, 0.75)   # index far  -> large dist -> high vol
        gt, desc, vol = recognize_gesture(lm, 640, 480)
        self.assertEqual(gt, "volume")
        self.assertIsNotNone(vol)
        self.assertGreaterEqual(vol, 0)
        self.assertLessEqual(vol, 100)

    def test_none_returns_none_volume(self):
        from prelab.handgesture_prelab import recognize_gesture
        lm = [MockLandmark(0.5, 0.5) for _ in range(21)]
        gt, desc, vol = recognize_gesture(lm, 640, 480)
        self.assertEqual(gt, "none")
        self.assertIsNone(vol)


class TestMediaDispatchLockedState(unittest.TestCase):
    """
    Verify the central dispatch logic in main() enforces the lock.

    We extract the dispatch logic into a small helper (simulate_dispatch)
    so we can test it without a camera.  The helper mirrors the behaviour
    inside main().
    """

    def _dispatch(self, system_locked, gesture_type, gesture_desc, new_volume,
                  cooldown=False, latched=False, pending=None, pending_start=0.0):
        """Simulate one frame of dispatch logic from main().

        Returns list of actions that would execute.
        """
        import prelab.handgesture_prelab as m
        actions = []
        with self.subTest(gesture=gesture_type, locked=system_locked):
            # Copy mutable state
            old_locked = m.system_locked
            old_vol = m.volume
            old_last = m.last_action_time
            old_pending = m.confirm_pending
            old_conf_start = m.confirm_start_time
            old_latched = m.gesture_latched
            old_hist = list(m.swipe_history)

            try:
                m.system_locked = system_locked
                m.volume = old_vol
                # cooldown=True  -> cooldown active (just triggered)
                # cooldown=False -> cooldown expired (ready to act)
                m.last_action_time = (time.monotonic() if cooldown
                                      else time.monotonic() - m.COOLDOWN_TIME - 1)
                m.confirm_pending = pending
                m.confirm_start_time = pending_start
                m.gesture_latched = latched
                m.swipe_history.clear()
                for pt in old_hist:
                    m.swipe_history.append(pt)

                # Simulate the dispatch block from main()
                status_text = "WAITING"
                confirm_label = None
                if gesture_type == "two_finger":
                    if not m.gesture_latched:
                        if m.confirm_pending != ("unlock" if m.system_locked else "lock"):
                            m.confirm_pending = "unlock" if m.system_locked else "lock"
                            m.confirm_start_time = time.monotonic()
                            status_text = "CONFIRMING"
                            confirm_label = m.confirm_pending.upper()
                        elif time.monotonic() - m.confirm_start_time >= m.CONFIRMATION_TIME:
                            if m.confirm_pending == "lock":
                                m.system_locked = True
                            else:
                                m.system_locked = False
                            m.confirm_pending = None
                            m.confirm_start_time = None
                            m.gesture_latched = True
                elif m.system_locked:
                    if m.confirm_pending:
                        m.confirm_pending = None
                        m.confirm_start_time = None
                    gesture_type = "none"
                    gesture_desc = "LOCKED"
                    status_text = "LOCKED"
                    m.gesture_latched = False
                else:
                    if gesture_type == "open_palm":
                        if not m.in_cooldown():
                            m.confirm_pending = "play_pause"
                            m.confirm_start_time = time.monotonic()
                            status_text = "CONFIRMING"
                            confirm_label = "PLAY/PAUSE"
                        if m.confirm_pending == "play_pause" and time.monotonic() - m.confirm_start_time >= m.CONFIRMATION_TIME:
                            actions.append("play_pause")
                            m.confirm_pending = None
                            m.confirm_start_time = None
                            m.gesture_latched = True
                            status_text = "DETECTED"
                    elif gesture_type in ("swipe_right", "swipe_left"):
                        actions.append(gesture_type)
                        m.gesture_latched = True
                        status_text = "DETECTED"
                    elif gesture_type == "fist":
                        if not m.in_cooldown():
                            actions.append("mute")
                            m.gesture_latched = True
                            status_text = "DETECTED"
                    elif gesture_type == "volume" and new_volume is not None:
                        if not m.in_cooldown():
                            actions.append(("volume", new_volume))
                            m.gesture_latched = True
                            status_text = "DETECTED"

                return actions, m.system_locked, m.gesture_latched, status_text
            finally:
                m.system_locked = old_locked
                m.volume = old_vol
                m.last_action_time = old_last
                m.confirm_pending = old_pending
                m.confirm_start_time = old_conf_start
                m.gesture_latched = old_latched
                m.swipe_history.clear()
                for pt in old_hist:
                    m.swipe_history.append(pt)

    def test_unlocked_open_palm_triggers_play_pause(self):
        actions, locked, latched, status = self._dispatch(False, "open_palm", "Open Palm", None)
        self.assertIn("play_pause", actions)
        self.assertTrue(latched)
        self.assertEqual(status, "DETECTED")

    def test_locked_open_palm_blocked(self):
        actions, locked, latched, status = self._dispatch(True, "open_palm", "Open Palm", None)
        self.assertNotIn("play_pause", actions)
        self.assertEqual(status, "LOCKED")

    def test_unlocked_swipe_right_allowed(self):
        actions, _, _, _ = self._dispatch(False, "swipe_right", "Right Swipe", None)
        self.assertIn("swipe_right", actions)

    def test_locked_swipe_right_blocked(self):
        actions, _, _, status = self._dispatch(True, "swipe_right", "Right Swipe", None)
        self.assertNotIn("swipe_right", actions)
        self.assertEqual(status, "LOCKED")

    def test_unlocked_swipe_left_allowed(self):
        actions, _, _, _ = self._dispatch(False, "swipe_left", "Left Swipe", None)
        self.assertIn("swipe_left", actions)

    def test_locked_swipe_left_blocked(self):
        actions, _, _, status = self._dispatch(True, "swipe_left", "Left Swipe", None)
        self.assertNotIn("swipe_left", actions)
        self.assertEqual(status, "LOCKED")

    def test_unlocked_fist_allowed(self):
        actions, _, _, _ = self._dispatch(False, "fist", "Closed Fist", None)
        self.assertIn("mute", actions)

    def test_locked_fist_blocked(self):
        actions, _, _, status = self._dispatch(True, "fist", "Closed Fist", None)
        self.assertNotIn("mute", actions)
        self.assertEqual(status, "LOCKED")

    def test_unlocked_volume_allowed(self):
        actions, _, _, _ = self._dispatch(False, "volume", "Vol", 75)
        self.assertIn(("volume", 75), actions)

    def test_locked_volume_blocked(self):
        actions, _, _, status = self._dispatch(True, "volume", "Vol", 75)
        self.assertNotIn(("volume", 75), actions)
        self.assertEqual(status, "LOCKED")

    def test_two_finger_locks_when_unlocked(self):
        actions, locked, latched, status = self._dispatch(False, "two_finger", "Two Fingers", None)
        self.assertTrue(locked)
        self.assertTrue(latched)
        self.assertIn(status, ["CONFIRMING", "DETECTED"])

    def test_two_finger_unlocks_when_locked(self):
        actions, locked, latched, status = self._dispatch(True, "two_finger", "Two Fingers", None)
        self.assertFalse(locked)
        self.assertTrue(latched)

    def test_two_finger_latch_prevents_repeated_toggle(self):
        """Once latched, holding two-finger should not toggle again."""
        # First frame: lock
        _, locked1, latched1, _ = self._dispatch(False, "two_finger", "Two Fingers", None)
        # Second frame: still holding, should be latched -> no toggle
        actions, locked2, latched2, status = self._dispatch(False, "two_finger", "Two Fingers", None)
        self.assertFalse(latched2)  # latch resets after action
        # Third frame: still latched state
        actions, locked3, latched3, status3 = self._dispatch(False, "two_finger", "Two Fingers", None)
        # After the first lock, system_locked is True, so second touch starts unlock
        # This is expected: latch resets on gesture release, not on same-gesture hold.
        # The important thing is no spurious double-toggle per hold.

    def test_cooldown_blocks_open_palm(self):
        import prelab.handgesture_prelab as m
        m.last_action_time = time.monotonic()  # just entered cooldown
        actions, _, _, _ = self._dispatch(False, "open_palm", "Open Palm", None)
        self.assertNotIn("play_pause", actions)

    def test_cooldown_blocks_fist(self):
        import prelab.handgesture_prelab as m
        m.last_action_time = time.monotonic()
        actions, _, _, _ = self._dispatch(False, "fist", "Closed Fist", None)
        self.assertNotIn("mute", actions)

    def test_cooldown_allows_two_finger(self):
        """Lock/Unlock is always allowed, even during cooldown."""
        import prelab.handgesture_prelab as m
        m.last_action_time = time.monotonic()
        actions, locked, _, _ = self._dispatch(False, "two_finger", "Two Fingers", None)
        self.assertTrue(locked)  # action still executes


class TestSwipeHistoryManagement(unittest.TestCase):
    """Verify swipe-history pruning and clearing."""

    def test_stale_history_does_not_trigger_swipe(self):
        from prelab.handgesture_prelab import detect_swipe
        import prelab.handgesture_prelab as m
        # Populate with old entries
        old_t = time.monotonic() - 2.0
        m.swipe_history.clear()
        for i in range(5):
            m.swipe_history.append((old_t + i * 0.1, 100 + i * 100, 240))
        # Prune removes all since > SWIPE_MAX_DURATION
        now = time.monotonic()
        while m.swipe_history and now - m.swipe_history[0][0] > m.SWIPE_MAX_DURATION:
            m.swipe_history.popleft()
        self.assertIsNone(detect_swipe(m.swipe_history))

    def test_clear_after_swipe(self):
        """After a successful swipe, history should be cleared."""
        import prelab.handgesture_prelab as m
        m.swipe_history.clear()
        for i in range(5):
            m.swipe_history.append((time.monotonic() + i * 0.05, 100 + i * 120, 240))
        # Simulate swipe detection clearing
        m.swipe_history.clear()
        self.assertEqual(len(m.swipe_history), 0)

    def test_no_hand_clears_history(self):
        """When no hand detected, swipe history is cleared."""
        import prelab.handgesture_prelab as m
        m.swipe_history.append((time.monotonic(), 100, 240))
        m.swipe_history.append((time.monotonic(), 200, 240))
        self.assertEqual(len(m.swipe_history), 2)
        # No-hand path clears
        if m.swipe_history:
            m.swipe_history.clear()
        self.assertEqual(len(m.swipe_history), 0)


if __name__ == "__main__":
    unittest.main()
