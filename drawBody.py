import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import Settings as S
from data import MeshConnections as MC

PoseLandmarker = vision.PoseLandmarker
BaseOptions = python.BaseOptions

# baseOptions = S.BODY_ACCURACY
# basePath = ""
# if baseOptions == 1:
#     basePath = "data/pose_landmarker_lite.task"
# elif baseOptions == 2:
#     basePath = "data/pose_landmarker_full.task"
# else:
#     basePath = "data/pose_landmarker_heavy.task"

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="data/pose_landmarker_full.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_connection_color(start, end):
    """Get color based on which side of body the connection is on."""
    conn = (start, end)
    if S.BODY_COLORED:
        if conn in MC.LEFT_CONNECTIONS:
            return S.BODY_LEFT_COLOR
        elif conn in MC.RIGHT_CONNECTIONS:
            return S.BODY_RIGHT_COLOR
        else:
            return S.BODY_LINE_COLOR
    return (255, 255, 255)


def get_landmark_color(idx):
    """Get color based on which side of body the landmark is on."""
    if S.BODY_COLORED:
        if idx in MC.LEFT_LANDMARKS:
            return S.BODY_LEFT_COLOR
        elif idx in MC.RIGHT_LANDMARKS:
            return S.BODY_RIGHT_COLOR
        else:
            return S.BODY_DOT_COLOR
    return (255, 255, 255)


def draw_body(frame, result):
    """Draw pose landmarks and connections."""
    if not result.pose_landmarks:
        return frame

    h, w, c = frame.shape

    for pose_landmarks in result.pose_landmarks:
        # Convert normalized landmarks to pixel coordinates
        points = []
        for lm in pose_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
            points.append((px, py, visibility))

        # Draw connections
        for start_idx, end_idx in MC.BODY_CONNECTIONS:
            if start_idx >= len(points) or end_idx >= len(points):
                continue

            p1 = points[start_idx]
            p2 = points[end_idx]

            # Skip if either point has low visibility
            if p1[2] < 0.5 or p2[2] < 0.5:
                continue

            color = get_connection_color(start_idx, end_idx)
            cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, 2, cv2.LINE_AA)

        # Draw landmarks (skip face landmarks 0-10 since we have separate face detection)
        for idx, (px, py, vis) in enumerate(points):
            if idx < 11:  # Skip face landmarks
                continue
            if vis < 0.5:
                continue

            color = get_landmark_color(idx)
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_body_full(frame, result):
    """Draw all pose landmarks including face points."""
    if not result.pose_landmarks:
        return frame

    h, w, c = frame.shape

    for pose_landmarks in result.pose_landmarks:
        points = []
        for lm in pose_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
            points.append((px, py, visibility))

        # Draw connections
        for start_idx, end_idx in MC.BODY_CONNECTIONS:
            if start_idx >= len(points) or end_idx >= len(points):
                continue

            p1 = points[start_idx]
            p2 = points[end_idx]

            if p1[2] < 0.5 or p2[2] < 0.5:
                continue

            color = get_connection_color(start_idx, end_idx)
            cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, 2, cv2.LINE_AA)

        # Draw ALL landmarks including face
        for idx, (px, py, vis) in enumerate(points):
            if vis < 0.5:
                continue

            color = get_landmark_color(idx)
            size = 3 if idx < 11 else 5
            cv2.circle(frame, (px, py), size, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), size, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def get_body_info(result):
    """Get text info about detected poses."""
    if not result.pose_landmarks:
        return []

    info = []
    for i, pose_landmarks in enumerate(result.pose_landmarks):
        # Count visible landmarks
        visible = sum(1 for lm in pose_landmarks
                     if hasattr(lm, 'visibility') and lm.visibility > 0.5)
        total = len(pose_landmarks)
        info.append(f"Body {i+1}: {visible}/{total} points")

    return info