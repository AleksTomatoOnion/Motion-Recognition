import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from data import MeshConnections as MC
import Settings as S

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='data/face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

FACE_CONTOURS = frozenset(MC.FACE_CONTOURS)
NOSE = frozenset(MC.NOSE)

_DEFAULT_COLOR = (255, 255, 255)

# Key landmark labels
_KEY_POINTS = {
    1: "Nose",
    33: "L Eye",
    263: "R Eye",
    61: "Mouth L",
    291: "Mouth R",
    199: "Chin",
    10: "Forehead",
}


def _get_contour_color():
    if S.FACE_COLORED:
        return S.FACE_CONTOUR_COLOR
    return _DEFAULT_COLOR


def _get_nose_color():
    if S.FACE_COLORED:
        return S.FACE_NOSE_COLOR
    return _DEFAULT_COLOR


def _get_mesh_color():
    if S.FACE_COLORED:
        return S.FACE_MESH_COLOR
    return _DEFAULT_COLOR


def _get_dot_color():
    if S.FACE_COLORED:
        return S.FACE_DOT_COLOR
    return _DEFAULT_COLOR


def draw_face(frame, detection_result):
    if not detection_result.face_landmarks:
        return frame

    h, w = frame.shape[:2]

    contour_color = _get_contour_color()
    nose_color = _get_nose_color()
    mesh_color = _get_mesh_color()
    dot_color = _get_dot_color()

    for face_landmarks in detection_result.face_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        num_points = len(points)

        # Draw face contours (eyes, lips, eyebrows, face oval, irises)
        for start_idx, end_idx in FACE_CONTOURS:
            if start_idx < num_points and end_idx < num_points:
                cv2.line(frame, points[start_idx], points[end_idx],
                         contour_color, S.FACE_CONTOUR_THICKNESS)

        # Draw nose
        for start_idx, end_idx in NOSE:
            if start_idx < num_points and end_idx < num_points:
                cv2.line(frame, points[start_idx], points[end_idx],
                         nose_color, S.FACE_NOSE_THICKNESS)

        # Draw full mesh (connect nearby landmarks for mocap look)
        # This creates the triangulated mesh effect
        pts = np.array(points[:468], dtype=np.int32)  # Exclude iris points
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in pts:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                subdiv.insert((int(p[0]), int(p[1])))

        triangles = subdiv.getTriangleList()
        for t in triangles:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            # Only draw if all points are inside the frame
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h and
                0 <= pt3[0] < w and 0 <= pt3[1] < h):
                cv2.line(frame, pt1, pt2, mesh_color, S.FACE_CONTOUR_THICKNESS)
                cv2.line(frame, pt2, pt3, mesh_color, S.FACE_CONTOUR_THICKNESS)
                cv2.line(frame, pt3, pt1, mesh_color, S.FACE_CONTOUR_THICKNESS)

        if S.SHOW_FACE_TEXT:
            for idx, label in _KEY_POINTS.items():
                if idx < num_points:
                    cv2.circle(frame, points[idx], S.FACE_DOT_RADIUS, dot_color, -1)
                    cv2.putText(frame, label,
                                (points[idx][0] + 5, points[idx][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                S.LABEL_INFO_COLOR, 1)

    return frame


def draw_face_optimized(frame, face_result):
    if not face_result.face_landmarks:
        return frame

    h, w = frame.shape[:2]

    contour_color = _get_contour_color()
    nose_color = _get_nose_color()
    dot_color = _get_dot_color()

    for face_landmarks in face_result.face_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        num = len(points)

        # REMOVED: Subdiv2D mesh triangulation (huge CPU drain)
        # Just draw dots and contour lines instead

        # Dots (draw fewer - every 3rd point)
        for i, p in enumerate(points):
            if i % 3 == 0:
                cv2.circle(frame, p, S.FACE_DOT_RADIUS, dot_color, -1)

        # Contours
        for s, e in FACE_CONTOURS:
            if s < num and e < num:
                cv2.line(frame, points[s], points[e], contour_color, S.FACE_CONTOUR_THICKNESS)

        # Nose
        for s, e in NOSE:
            if s < num and e < num:
                cv2.line(frame, points[s], points[e], nose_color, S.FACE_NOSE_THICKNESS)

    return frame