"""
temporal_aggregator.py
State machine managing chronological gloss accumulation, confidence filtering,
spatial neutral-zone reset debouncing, transition epenthesis rejection,
and interactive user editing (insert, edit, delete, clear).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
import math
import numpy as np
import time

from landmarks import (
    LEFT_HAND_START,
    RIGHT_HAND_START,
    POSE_START,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    hands_missing,
)

@dataclass
class GlossEntry:
    id: int
    sign: str
    confidence: float
    timestamp: float
    duration_ms: float
    status: Literal['accepted', 'low_confidence', 'edited', 'discarded']
    origin: Literal['recognized', 'manual', 'edited'] = 'recognized'
    top3: List[Dict[str, Any]] = field(default_factory=list)
    end_reason: str = "absent"

class TemporalAggregator:
    def __init__(
        self,
        min_confidence: float = 0.35,
        hold_debounce_ms: float = 800.0,
        epenthesis_window_ms: float = 400.0
    ):
        self.min_confidence = min_confidence
        self.hold_debounce_ms = hold_debounce_ms
        self.epenthesis_window_ms = epenthesis_window_ms

        self.entries: List[GlossEntry] = []
        self._next_id: int = 1
        self.last_accepted_sign: Optional[str] = None
        self.last_accepted_time: Optional[float] = None
        self.has_neutral_reset: bool = True
        self.last_hands_absent_time: Optional[float] = None
        self.diagnostic_history: List[Dict[str, Any]] = []

    def reset(self):
        self.entries.clear()
        self._next_id = 1
        self.last_accepted_sign = None
        self.last_accepted_time = None
        self.has_neutral_reset = True
        self.last_hands_absent_time = None
        self.diagnostic_history.clear()

    def get_confirmed_glosses(self) -> List[str]:
        return [e.sign.upper() for e in self.entries if e.status in ('accepted', 'edited')]

    def get_entries(self) -> List[GlossEntry]:
        return list(self.entries)

    # --- Spatial Neutral-Zone Tracking ---
    def notify_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Evaluates real-time landmark coordinates to detect if hands returned to neutral rest position
        (below shoulders or absent from camera view).
        Layout: 543 * 3 float32 coordinates.
        """
        now = timestamp if timestamp is not None else time.time() * 1000.0

        if hands_missing(frame):
            if self.last_hands_absent_time is None:
                self.last_hands_absent_time = now
            # Hands absent for >= 350ms (strictly greater than segmenter gap_ms 300ms) triggers neutral reset
            # This ensures that a segment ending with 'absent' is fully closed and processed BEFORE neutral reset activates
            if (now - self.last_hands_absent_time) >= 350.0:
                self.has_neutral_reset = True
            return
        else:
            self.last_hands_absent_time = None

        # Extract shoulders (pose landmarks 11 and 12)
        l_idx = (POSE_START + POSE_LEFT_SHOULDER) * 3
        r_idx = (POSE_START + POSE_RIGHT_SHOULDER) * 3
        lx, ly = frame[l_idx], frame[l_idx + 1]
        rx, ry = frame[r_idx], frame[r_idx + 1]

        shoulders_valid = np.isfinite(lx) and np.isfinite(ly) and np.isfinite(rx) and np.isfinite(ry)

        # Extract hand mesh wrists (landmarks 468 and 522)
        hw_l_idx = LEFT_HAND_START * 3
        hw_r_idx = RIGHT_HAND_START * 3
        hw_ly = frame[hw_l_idx + 1]
        hw_ry = frame[hw_r_idx + 1]

        # Extract pose skeleton wrists (landmarks 504 and 505)
        pw_l_idx = (POSE_START + 15) * 3 # POSE_LEFT_WRIST = 15
        pw_r_idx = (POSE_START + 16) * 3 # POSE_RIGHT_WRIST = 16
        pw_ly = frame[pw_l_idx + 1]
        pw_ry = frame[pw_r_idx + 1]

        # Conservatively take the higher elevation (smaller y) between hand mesh wrist and pose skeleton wrist
        left_ys = [y for y in [hw_ly, pw_ly] if np.isfinite(y)]
        right_ys = [y for y in [hw_ry, pw_ry] if np.isfinite(y)]

        lw_valid = len(left_ys) > 0
        rw_valid = len(right_ys) > 0
        lwy = min(left_ys) if lw_valid else np.nan
        rwy = min(right_ys) if rw_valid else np.nan

        if not lw_valid and not rw_valid:
            self.has_neutral_reset = True
            return

        if shoulders_valid:
            cy = (ly + ry) / 2.0
            scale = max(math.hypot(lx - rx, ly - ry), 0.1)
            # Neutral zone threshold: 0.15 * shoulder_width below shoulder midpoint
            neutral_y = cy + (0.15 * scale)

            lw_neutral = (not lw_valid) or (lwy > neutral_y)
            rw_neutral = (not rw_valid) or (rwy > neutral_y)

            if lw_neutral and rw_neutral:
                self.has_neutral_reset = True
        else:
            # Fallback if body landmarks are unavailable (e.g. shoulders cropped):
            # If wrist y is in the bottom 40% of the frame (y > 0.60 in normalized coords)
            lw_bot = (not lw_valid) or (lwy > 0.60)
            rw_bot = (not rw_valid) or (rwy > 0.60)
            if lw_bot and rw_bot:
                self.has_neutral_reset = True

    def notify_neutral_reset(self) -> None:
        """Explicit signal that hands have returned to neutral rest position."""
        self.has_neutral_reset = True

    # --- Recognition Processing ---
    def process_recognition(
        self,
        recognition: Dict[str, Any],
        end_reason: str = "absent",
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a new isolated sign recognition event from the segmenter.
        Applies spatial neutral-zone debouncing, transition epenthesis filtering, and confidence thresholding.
        Handles both 'still' and 'absent' end reasons without blind window expansion.
        """
        now = timestamp if timestamp is not None else time.time() * 1000.0
        sign = recognition["sign"]
        confidence = recognition["confidence"]
        duration_ms = recognition.get("duration_ms", 0.0)
        top3 = recognition.get("top3", [])

        # 1. Debounce duplicate identical sign without neutral reset
        # Handles BOTH end_reason == 'still' (hold) and end_reason == 'absent' (hand lowering exit)
        if (
            self.last_accepted_sign == sign
            and self.last_accepted_time is not None
            and (now - self.last_accepted_time) < self.hold_debounce_ms
        ):
            if not self.has_neutral_reset:
                # Update duration of the last accepted entry rather than inserting a duplicate
                if self.entries and self.entries[-1].status == "accepted":
                    self.entries[-1].duration_ms += duration_ms
                diag = {
                    "action": "debounced",
                    "sign": sign,
                    "confidence": confidence,
                    "timestamp": now,
                    "end_reason": end_reason,
                    "reason": "identical_sign_without_neutral_reset"
                }
                self.diagnostic_history.append(diag)
                return {"status": "debounced", "entry": self.entries[-1] if self.entries else None, "diagnostic": diag}

        # 2. Filter Transitional Movement Epenthesis
        # If a different sign arrives very rapidly (< 400ms) after an accepted sign without neutral reset
        # and has marginal confidence (< 0.50), reject as transitional hand movement
        if (
            self.last_accepted_time is not None
            and (now - self.last_accepted_time) < self.epenthesis_window_ms
            and not self.has_neutral_reset
            and confidence < 0.50
        ):
            diag = {
                "action": "debounced",
                "sign": sign,
                "confidence": confidence,
                "timestamp": now,
                "end_reason": end_reason,
                "reason": "transition_epenthesis_suppressed"
            }
            self.diagnostic_history.append(diag)
            return {"status": "debounced", "entry": None, "diagnostic": diag}

        # 3. Check Confidence Filtering
        if confidence < self.min_confidence:
            entry = GlossEntry(
                id=self._next_id,
                sign=sign,
                confidence=confidence,
                timestamp=now,
                duration_ms=duration_ms,
                status="low_confidence",
                origin="recognized",
                top3=top3,
                end_reason=end_reason
            )
            self._next_id += 1
            self.entries.append(entry)
            diag = {
                "action": "flagged_low_confidence",
                "sign": sign,
                "confidence": confidence,
                "threshold": self.min_confidence,
                "timestamp": now
            }
            self.diagnostic_history.append(diag)
            return {"status": "low_confidence", "entry": entry, "diagnostic": diag}

        # 4. Accepted Distinct Sign
        entry = GlossEntry(
            id=self._next_id,
            sign=sign.upper(),
            confidence=confidence,
            timestamp=now,
            duration_ms=duration_ms,
            status="accepted",
            origin="recognized",
            top3=top3,
            end_reason=end_reason
        )
        self._next_id += 1
        self.entries.append(entry)
        self.last_accepted_sign = sign
        self.last_accepted_time = now
        self.has_neutral_reset = False # Require neutral reset before duplicate can be accepted

        diag = {
            "action": "accepted",
            "sign": sign,
            "confidence": confidence,
            "timestamp": now,
            "end_reason": end_reason
        }
        self.diagnostic_history.append(diag)
        return {"status": "accepted", "entry": entry, "diagnostic": diag}

    # --- Interactive Buffer Operations ---
    def add_manual_gloss(self, gloss: str) -> Optional[GlossEntry]:
        trimmed = gloss.strip().upper()
        if not trimmed: return None
        entry = GlossEntry(
            id=self._next_id,
            sign=trimmed,
            confidence=1.0,
            timestamp=time.time() * 1000.0,
            duration_ms=0.0,
            status="accepted",
            origin="manual",
            end_reason="manual"
        )
        self._next_id += 1
        self.entries.append(entry)
        self.has_neutral_reset = True
        return entry

    def edit_gloss(self, entry_id: int, new_gloss: str) -> bool:
        trimmed = new_gloss.strip().upper()
        if not trimmed: return False
        for e in self.entries:
            if e.id == entry_id:
                e.sign = trimmed
                e.status = "edited"
                e.origin = "edited"
                return True
        return False

    def delete_gloss(self, entry_id: int) -> bool:
        prev_len = len(self.entries)
        self.entries = [e for e in self.entries if e.id != entry_id]
        return len(self.entries) < prev_len

    def delete_last(self) -> bool:
        if self.entries:
            self.entries.pop()
            return True
        return False

    def clear(self) -> None:
        self.entries.clear()
        self.last_accepted_sign = None
        self.last_accepted_time = None
        self.has_neutral_reset = True
