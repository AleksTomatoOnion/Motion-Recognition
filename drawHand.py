import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from data import MeshConnections as MC
import Settings as S

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='data/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark names
LANDMARK_NAMES = {
    0: "Wrist",
    4: "Thumb Tip",
    8: "Index Tip",
    12: "Middle Tip",
    16: "Ring Tip",
    20: "Pinky Tip",
}

FINGER_DEFS = [
    ([1, 2, 3, 4],     "HAND_THUMB_COLOR",  1.3),
    ([5, 6, 7, 8],     "HAND_INDEX_COLOR",  1.0),
    ([9, 10, 11, 12],  "HAND_MIDDLE_COLOR", 1.0),
    ([13, 14, 15, 16], "HAND_RING_COLOR",   0.9),
    ([17, 18, 19, 20], "HAND_PINKY_COLOR",  0.8),
]

# Precomputed landmark -> finger color attribute lookup
_LANDMARK_COLOR_MAP = {}
for _indices, _color_name, _ in FINGER_DEFS:
    for _idx in _indices:
        _LANDMARK_COLOR_MAP[_idx] = _color_name

# Precomputed connection sets (include wrist=0 in each finger group)
_CONNECTION_SETS = [
    (frozenset({0, 1, 2, 3, 4}),      "HAND_THUMB_COLOR"),
    (frozenset({0, 5, 6, 7, 8}),      "HAND_INDEX_COLOR"),
    (frozenset({0, 9, 10, 11, 12}),   "HAND_MIDDLE_COLOR"),
    (frozenset({0, 13, 14, 15, 16}),  "HAND_RING_COLOR"),
    (frozenset({0, 17, 18, 19, 20}),  "HAND_PINKY_COLOR"),
]

PALM_TRIANGLES = [
    (0, 1, 5), (0, 17, 13), (0, 5, 9), (0, 9, 13),
    (5, 9, 6), (9, 13, 10), (13, 17, 14),
    (5, 6, 9), (6, 9, 10), (9, 10, 13),
    (10, 13, 14), (13, 14, 17), (14, 17, 18),
    (1, 2, 5), (2, 5, 6),
]

_DEFAULT_COLOR = (255, 255, 255)
_FINGERTIP_SET = frozenset({4, 8, 12, 16, 20})


def _get_perpendicular_offset(p1, p2, width):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    nx = -dy / length * width
    ny = dx / length * width
    return int(nx), int(ny)


def _build_finger_mesh(points, finger_indices, base_width=8):
    n = len(finger_indices)
    if n < 2:
        return []

    left_points = []
    right_points = []
    center_points = []

    for i in range(n):
        idx = finger_indices[i]
        cx, cy = points[idx]
        center_points.append((cx, cy))

        taper = 1.0 - (i / n) * 0.4
        width = base_width * taper

        if i < n - 1:
            nx, ny = _get_perpendicular_offset(points[idx], points[finger_indices[i + 1]], width)
        else:
            nx, ny = _get_perpendicular_offset(points[finger_indices[i - 1]], points[idx], width)

        left_points.append((cx + nx, cy + ny))
        right_points.append((cx - nx, cy - ny))

    triangles = []
    for i in range(n - 1):
        lp_i, lp_n = left_points[i], left_points[i + 1]
        rp_i, rp_n = right_points[i], right_points[i + 1]
        cp_i, cp_n = center_points[i], center_points[i + 1]
        triangles.append((lp_i, lp_n, cp_i))
        triangles.append((rp_i, rp_n, cp_i))
        triangles.append((lp_n, cp_i, cp_n))
        triangles.append((rp_n, cp_i, cp_n))

    last = n - 1
    triangles.append((left_points[last], right_points[last], center_points[last]))
    return triangles


def get_connection_color(start, end):
    if not S.HAND_COLORED:
        return _DEFAULT_COLOR
    for conn_set, color_name in _CONNECTION_SETS:
        if start in conn_set and end in conn_set:
            return getattr(S, color_name)
    return S.HAND_PALM_COLOR


def _get_joint_color(idx):
    if not S.HAND_COLORED:
        return _DEFAULT_COLOR
    color_name = _LANDMARK_COLOR_MAP.get(idx)
    if color_name:
        return getattr(S, color_name)
    return S.HAND_DOT_COLOR


def _get_finger_color(color_name):
    if S.HAND_COLORED:
        return getattr(S, color_name)
    return _DEFAULT_COLOR


def _darken_color(color, amount=50):
    return (max(0, color[0] - amount), max(0, color[1] - amount), max(0, color[2] - amount))


# ============================================================
#                    DRAWING FUNCTIONS
# ============================================================

def _draw_wireframe(frame, points):
    """Draw mesh wireframe with triangle edges (no fill)."""
    if len(points) < 21:
        return frame

    palm_color = get_connection_color(-1, -1)

    for i0, i1, i2 in PALM_TRIANGLES:
        cv2.line(frame, points[i0], points[i1], palm_color, 1, cv2.LINE_AA)
        cv2.line(frame, points[i1], points[i2], palm_color, 1, cv2.LINE_AA)
        cv2.line(frame, points[i2], points[i0], palm_color, 1, cv2.LINE_AA)

    for finger_indices, color_name, width_mult in FINGER_DEFS:
        color = _get_finger_color(color_name)
        base_width = int(S.HAND_MESH_WIDTH * width_mult)
        triangles = _build_finger_mesh(points, finger_indices, base_width)
        for p1, p2, p3 in triangles:
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p3, color, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p1, color, 1, cv2.LINE_AA)

    dot_color = get_connection_color(-1, -1)
    for pt in points:
        cv2.circle(frame, pt, 3, dot_color, -1, cv2.LINE_AA)

    return frame


def _draw_wireframe_colored(frame, points):
    """Draw mesh wireframe with optional fill (S.HAND_FILLED) and optional colors (S.HAND_COLORED)."""
    if len(points) < 21:
        return frame

    # ------- FILL (controlled by S.HAND_FILLED, NOT S.HAND_COLORED) -------
    if S.HAND_FILLED:
        overlay = frame.copy()

        palm_fill_color = get_connection_color(-1, -1)
        for i0, i1, i2 in PALM_TRIANGLES:
            triangle = np.array([points[i0], points[i1], points[i2]], dtype=np.int32)
            cv2.fillPoly(overlay, [triangle], palm_fill_color)

        for finger_indices, color_name, width_mult in FINGER_DEFS:
            color = _get_finger_color(color_name)
            base_width = int(S.HAND_MESH_WIDTH * width_mult)
            triangles = _build_finger_mesh(points, finger_indices, base_width)
            for tri_pts in triangles:
                triangle = np.array(tri_pts, dtype=np.int32)
                cv2.fillPoly(overlay, [triangle], color)

        cv2.addWeighted(overlay, S.HAND_WIREFRAME_OPACITY, frame,
                        1.0 - S.HAND_WIREFRAME_OPACITY, 0, frame)

    # ------- PALM WIREFRAME EDGES -------
    palm_edge_color = _darken_color(S.HAND_PALM_COLOR if S.HAND_COLORED else _DEFAULT_COLOR)
    for i0, i1, i2 in PALM_TRIANGLES:
        cv2.line(frame, points[i0], points[i1], palm_edge_color, 1, cv2.LINE_AA)
        cv2.line(frame, points[i1], points[i2], palm_edge_color, 1, cv2.LINE_AA)
        cv2.line(frame, points[i2], points[i0], palm_edge_color, 1, cv2.LINE_AA)

    # ------- FINGER MESH EDGES -------
    for finger_indices, color_name, width_mult in FINGER_DEFS:
        color = _get_finger_color(color_name)
        edge_color = _darken_color(color)
        base_width = int(S.HAND_MESH_WIDTH * width_mult)
        triangles = _build_finger_mesh(points, finger_indices, base_width)
        for p1, p2, p3 in triangles:
            cv2.line(frame, p1, p2, edge_color, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p3, edge_color, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p1, edge_color, 1, cv2.LINE_AA)

    # ------- BONES (center lines) -------
    for start_idx, end_idx in MC.HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            color = get_connection_color(start_idx, end_idx)
            cv2.line(frame, points[start_idx], points[end_idx],
                     color, S.HAND_LINE_THICKNESS, cv2.LINE_AA)

    # ------- JOINTS -------
    dot_r = S.HAND_DOT_RADIUS
    for idx, pt in enumerate(points):
        color = _get_joint_color(idx)
        if idx in _FINGERTIP_SET:
            r = dot_r
        elif idx == 0:
            r = dot_r + 2
        else:
            r = dot_r - 1
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_hands(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame

    h, w = frame.shape[:2]

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[i][0].category_name

        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        num_points = len(points)

        # ------- CHOOSE DRAWING MODE -------
        if S.HAND_OPTIMIZED:
            for start_idx, end_idx in MC.HAND_CONNECTIONS:
                if start_idx < num_points and end_idx < num_points:
                    color = get_connection_color(start_idx, end_idx)
                    cv2.line(frame, points[start_idx], points[end_idx],
                             color, S.HAND_LINE_THICKNESS)

            for idx, point in enumerate(points):
                dot_color = get_connection_color(-1, -1)
                if idx in _FINGERTIP_SET:
                    cv2.circle(frame, point, S.HAND_DOT_RADIUS, dot_color, -1)
                    cv2.circle(frame, point, S.HAND_DOT_RADIUS - 1, (0, 0, 0), -1)
                elif idx == 0:
                    cv2.circle(frame, point, S.HAND_DOT_RADIUS + 3, dot_color, -1)
                else:
                    cv2.circle(frame, point, S.HAND_DOT_RADIUS, dot_color, -1)
        else:       
            frame = _draw_wireframe_colored(frame, points)


        # ------- TEXT LABELS -------
        if S.SHOW_HAND_TEXT:
            for idx, label in LANDMARK_NAMES.items():
                if idx < num_points:
                    cv2.putText(frame, label,
                                (points[idx][0] + 10, points[idx][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                S.LABEL_INFO_COLOR, 1)
            if num_points > 0:
                cv2.putText(frame, handedness,
                            (points[0][0] - 30, points[0][1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            S.HAND_BOX_COLOR, 2)

        # ------- BOUNDING BOX -------
        if S.DRAW_HAND_BOXES:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            padding = 20
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_max = min(h, max(y_coords) + padding)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          S.HAND_BOX_COLOR, S.HAND_BOX_THICKNESS)

    return frame


def count_fingers(hand_landmarks, handedness="Right"):
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    fingers_up = 0

    actual_hand = "Left" if handedness == "Right" else "Right"

    wrist = points[0]
    index_mcp = points[5]
    pinky_mcp = points[17]

    if actual_hand == "Right":
        palm_by_position = pinky_mcp[0] < index_mcp[0]
    else:
        palm_by_position = pinky_mcp[0] > index_mcp[0]

    palm_center_x = (points[0][0] + points[5][0] + points[17][0]) / 3.0
    palm_center_y = (points[0][1] + points[5][1] + points[17][1]) / 3.0

    thumb_tip_dist = ((points[4][0] - palm_center_x) ** 2 +
                      (points[4][1] - palm_center_y) ** 2) ** 0.5
    thumb_base_dist = ((points[2][0] - palm_center_x) ** 2 +
                       (points[2][1] - palm_center_y) ** 2) ** 0.5

    if thumb_tip_dist > thumb_base_dist * 1.2:
        thumb_to_index = ((points[4][0] - points[8][0]) ** 2 +
                          (points[4][1] - points[8][1]) ** 2) ** 0.5
        index_to_middle = ((points[8][0] - points[12][0]) ** 2 +
                           (points[8][1] - points[12][1]) ** 2) ** 0.5
        if thumb_to_index > index_to_middle * 0.5:
            fingers_up += 1

    finger_tips = (8, 12, 16, 20)
    finger_pips = (6, 10, 14, 18)
    finger_mcps = (5, 9, 13, 17)

    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        tip_above_pip = points[tip][1] < points[pip][1]

        tip_to_wrist = ((points[tip][0] - wrist[0]) ** 2 +
                        (points[tip][1] - wrist[1]) ** 2) ** 0.5
        pip_to_wrist = ((points[pip][0] - wrist[0]) ** 2 +
                        (points[pip][1] - wrist[1]) ** 2) ** 0.5

        tip_to_mcp = ((points[tip][0] - points[mcp][0]) ** 2 +
                      (points[tip][1] - points[mcp][1]) ** 2) ** 0.5
        pip_to_mcp = ((points[pip][0] - points[mcp][0]) ** 2 +
                      (points[pip][1] - points[mcp][1]) ** 2) ** 0.5

        checks_passed = 0
        if tip_above_pip:
            checks_passed += 1
        if tip_to_wrist > pip_to_wrist:
            checks_passed += 1
        if tip_to_mcp > pip_to_mcp * 0.8:
            checks_passed += 1

        if checks_passed >= 2:
            fingers_up += 1

    return fingers_up