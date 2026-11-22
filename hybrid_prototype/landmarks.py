import math
import numpy as np

# Landmark topology constants matching MediaPipe Holistic / asl-signs layout
N_FACE = 468
N_HAND = 21
N_POSE = 33
N_POINTS = N_FACE + N_HAND + N_POSE + N_HAND # 543
FRAME_LENGTH = N_POINTS * 3 # 1629

LEFT_HAND_START = N_FACE # 468
POSE_START = LEFT_HAND_START + N_HAND # 489
RIGHT_HAND_START = POSE_START + N_POSE # 522

POSE_UPPER = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

LIP_RUNS = [
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    [185, 40, 39, 37, 0, 267, 269, 270, 409],
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    [191, 80, 81, 82, 13, 312, 311, 310, 415],
]
LIPS = [p for run in LIP_RUNS for p in run]

HAND_WRIST = 0
HAND_MIDDLE_MCP = 9

def empty_frame():
    """Create a blank frame of 543*3 elements initialized to NaN."""
    return np.full(FRAME_LENGTH, np.nan, dtype=np.float32)

def hands_missing(frame: np.ndarray) -> bool:
    """True if neither hand has valid coordinates."""
    lh_valid = np.isfinite(frame[LEFT_HAND_START * 3])
    rh_valid = np.isfinite(frame[RIGHT_HAND_START * 3])
    return not (lh_valid or rh_valid)

