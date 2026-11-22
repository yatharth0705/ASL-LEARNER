import os
import sys
import time
from typing import Dict, Any, List, Optional
import numpy as np

from landmarks import empty_frame, FRAME_LENGTH
from segmenter import Segmenter, SegmenterOptions, Segment
from recognizer import GestureSignRecognizer
from temporal_aggregator import TemporalAggregator, GlossEntry
from translator import GlossaBARTTranslator

class HybridASLPipeline:
    def __init__(
        self,
        min_confidence: float = 0.35,
        hold_debounce_ms: float = 900.0,
        segmenter_opts: Optional[SegmenterOptions] = None,
        load_translator: bool = True
    ):
        self.segmenter = Segmenter(segmenter_opts or SegmenterOptions())
        self.recognizer = GestureSignRecognizer()
        self.aggregator = TemporalAggregator(
            min_confidence=min_confidence,
            hold_debounce_ms=hold_debounce_ms
        )
        self.translator = GlossaBARTTranslator() if load_translator else None
        self.telemetry_log: List[Dict[str, Any]] = []

    def reset(self):
        self.segmenter.reset()
        self.aggregator.reset()
        self.telemetry_log.clear()

    def push_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        poses: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Push a single frame of 543*3 coordinates (MediaPipe Holistic layout).
        Returns a result dictionary if a sign was recognized on this frame, else None.
        """
        # Update spatial neutral-zone tracker on aggregator with every frame
        self.aggregator.notify_frame(frame, timestamp_ms)

        event = self.segmenter.push(frame, timestamp_ms, crowded=(poses > 1))
        if not event:
            return None

        event_type = event["type"]
        if event_type == "start":
            entry = {"event": "start", "timestamp": timestamp_ms}
            self.telemetry_log.append(entry)
            return entry

        if event_type == "discard":
            entry = {
                "event": "discard",
                "reason": event["reason"],
                "timestamp": timestamp_ms
            }
            self.telemetry_log.append(entry)
            return entry

        if event_type == "end":
            reason = event["reason"]
            segment: Segment = event["segment"]

            # Run GestureSign Recognition
            t_rec_start = time.perf_counter()
            recognition = self.recognizer.classify_segment(segment)
            rec_ms = (time.perf_counter() - t_rec_start) * 1000.0

            # Process through Temporal Aggregator
            agg_result = self.aggregator.process_recognition(
                recognition=recognition,
                end_reason=reason,
                timestamp=timestamp_ms
            )

            result_payload = {
                "event": "recognition",
                "end_reason": reason,
                "recognition": recognition,
                "aggregator_status": agg_result["status"],
                "current_buffer": self.aggregator.get_confirmed_glosses(),
                "duration_ms": segment.duration_ms,
                "raw_frames": len(segment.frames),
                "fps": segment.fps,
                "recognition_time_ms": rec_ms,
                "timestamp": timestamp_ms
            }
            self.telemetry_log.append(result_payload)
            return result_payload

        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        """Flush an open segment (e.g. at the end of video playback or user pauses)."""
        event = self.segmenter.flush()
        if not event:
            return None
        if event["type"] == "end":
            segment = event["segment"]
            recognition = self.recognizer.classify_segment(segment)
            agg_result = self.aggregator.process_recognition(
                recognition=recognition,
                end_reason="flush",
                timestamp=segment.timestamps[-1] if segment.timestamps else time.time()*1000
            )
            payload = {
                "event": "recognition",
                "end_reason": "flush",
                "recognition": recognition,
                "aggregator_status": agg_result["status"],
                "current_buffer": self.aggregator.get_confirmed_glosses(),
                "duration_ms": segment.duration_ms,
                "raw_frames": len(segment.frames),
                "fps": segment.fps
            }
            self.telemetry_log.append(payload)
            return payload
        return None

    def translate_buffer(self, use_beam_search: bool = False, num_beams: int = 4) -> Dict[str, Any]:
        """Translates the currently accumulated confirmed glosses into English."""
        if not self.translator:
            raise RuntimeError("Translator model was not loaded")
        glosses = self.aggregator.get_confirmed_glosses()
        return self.translator.translate(glosses, use_beam_search=use_beam_search, num_beams=num_beams)

    # Buffer Editing Operations (Delegated to Aggregator)
    def delete_last_gloss(self) -> Optional[GlossEntry]:
        return self.aggregator.delete_last()

    def delete_gloss_at(self, index: int) -> Optional[GlossEntry]:
        return self.aggregator.delete_at(index)

    def replace_gloss(self, index: int, new_sign: str) -> bool:
        return self.aggregator.replace_gloss(index, new_sign)

    def add_manual_gloss(self, sign: str) -> GlossEntry:
        return self.aggregator.add_manual_gloss(sign)

    def clear_buffer(self):
        self.aggregator.clear()

    def get_gloss_tokens(self) -> List[str]:
        return self.aggregator.get_confirmed_glosses()

    def get_all_entries(self) -> List[GlossEntry]:
        return self.aggregator.get_all_entries()

