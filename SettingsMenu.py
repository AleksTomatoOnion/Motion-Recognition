import cv2
import numpy as np
import Settings as S
import math

# ------- WINDOW CONFIG -------
WINDOW_NAME = "Settings"
WINDOW_WIDTH = 380
VISIBLE_HEIGHT = 500
ROW_HEIGHT = 40
PADDING = 15
HEADER_HEIGHT = 45
SCROLLBAR_WIDTH = 12

# Colors
BG_COLOR = (30, 30, 30)
SECTION_COLOR = (45, 45, 45)
TEXT_COLOR = (220, 220, 220)
VALUE_ON = (0, 200, 0)
VALUE_OFF = (0, 0, 200)
BTN_COLOR = (70, 70, 70)
BTN_HOVER = (100, 100, 100)
ACCENT = (0, 200, 200)
BORDER = (80, 80, 80)
SCROLLBAR_BG = (50, 50, 50)
SCROLLBAR_FG = (120, 120, 120)
SCROLLBAR_HOVER = (160, 160, 160)

# Layout
SETTINGS_LAYOUT = [
    ("VIEW",                   None,                "section", 0, 0, 0),
    ("Target FPS",             "TARGET_FPS",         "int",    5, 60, 1),
    ("Process Every N",        "PROCESS_EVERY_N",    "int",    1, 10, 1),
    # ("Frame Width",            "FRAME_WIDTH",        "int",  320, 1920, 160),
    # ("Frame Height",           "FRAME_HEIGHT",       "int",  240, 1080, 120),
    ("Show FPS",               "SHOW_FPS",           "bool",   0, 0, 0),
    ("Show Face Count",        "SHOW_FACE_COUNT",    "bool",   0, 0, 0),
    ("Show Finger Count",      "SHOW_FINGER_COUNT",  "bool",   0, 0, 0),
    ("Hand Text",              "SHOW_HAND_TEXT",      "bool",   0, 0, 0),
    ("Face Text",              "SHOW_FACE_TEXT",      "bool",   0, 0, 0),
    ("Body Text",              "SHOW_BODY_TEXT",      "bool",   0, 0, 0),
    ("Show Video",             "SHOW_VIDEO",         "bool",   0, 0, 0),

    ("DETECTION",              None,                "section", 0, 0, 0),
    ("Detect Face",            "DETECT_FACE",        "bool",   0, 0, 0),
    ("Detect Hands",           "DETECT_HANDS",       "bool",   0, 0, 0),
    ("Detect Body",            "DETECT_BODY",        "bool",   0, 0, 0),

    ("DRAWING",                None,                "section", 0, 0, 0),
    ("Draw Face",              "DRAW_FACE",          "bool",   0, 0, 0),
    ("Draw Hands",             "DRAW_HANDS",         "bool",   0, 0, 0),
    ("Draw Body",              "DRAW_BODY",          "bool",   0, 0, 0),
    ("Hand Boxes",             "DRAW_HAND_BOXES",    "bool",   0, 0, 0),
    ("Finger Line Thickness",  "HAND_LINE_THICKNESS", "int", 1, 10, 1),
    ("Hand Box Thickness",     "HAND_BOX_THICKNESS",  "int", 1, 10, 1),
    ("Hand Joint Radius",      "HAND_DOT_RADIUS",     "int", 1, 10, 1),
    ("Face Dot Radius",        "FACE_DOT_RADIUS",     "int", 1, 5, 1),
    ("Face Contour Thickness", "FACE_CONTOUR_THICKNESS", "int", 1, 10, 1),
    ("Nose Contour Thickness", "FACE_NOSE_THICKNESS",    "int", 1, 10, 1),

    ("OPTIMIZATION",           None,                "section", 0, 0, 0),
    ("Face Optimized",         "FACE_OPTIMIZED",    "bool",    0, 0, 0),
    ("Hand Optimized",         "HAND_OPTIMIZED",    "bool",    0, 0, 0),
    # ("Body Accuracy",          "BODY_ACCURACY",     "int", 1, 3, 1),

    ("FACE COLORS",            None,                "section", 0, 0, 0),
    ("Color Face",             "FACE_COLORED",       "bool",   0, 0, 0),
    ("Face Mesh",              "FACE_MESH_COLOR",    "color",  0, 0, 0),
    ("Face Contour",           "FACE_CONTOUR_COLOR", "color",  0, 0, 0),
    ("Face Nose",              "FACE_NOSE_COLOR",    "color",  0, 0, 0),
    ("Face Dots",              "FACE_DOT_COLOR",     "color",  0, 0, 0),

    ("HAND COLORS",            None,                "section", 0, 0, 0),
    ("Color Hands",            "HAND_COLORED",       "bool",  0, 0, 0),
    ("Fill Hands",             "HAND_FILLED",        "bool",  0, 0, 0),
    ("Wireframe Opacity",      "HAND_WIREFRAME_OPACITY", "int", 0, 100, 5),
    ("Thumb",                  "HAND_THUMB_COLOR",   "color",  0, 0, 0),
    ("Index",                  "HAND_INDEX_COLOR",   "color",  0, 0, 0),
    ("Middle",                 "HAND_MIDDLE_COLOR",  "color",  0, 0, 0),
    ("Ring",                   "HAND_RING_COLOR",    "color",  0, 0, 0),
    ("Pinky",                  "HAND_PINKY_COLOR",   "color",  0, 0, 0),
    ("Palm",                   "HAND_PALM_COLOR",    "color",  0, 0, 0),
    ("Hand Joint",             "HAND_DOT_COLOR",     "color",  0, 0, 0),
    ("Hand Box",               "HAND_BOX_COLOR",     "color",  0, 0, 0),
    
    ("BODY COLORS",            None,                "section", 0, 0, 0),
    ("Color Body",             "BODY_COLORED",       "bool",   0, 0, 0),
    ("Body Lines",             "BODY_LINE_COLOR",    "color",  0, 0, 0),
    ("Body Joints",            "BODY_DOT_COLOR",     "color",  0, 0, 0),
    ("Body Left",              "BODY_LEFT_COLOR",    "color",  0, 0, 0),
    ("Body Right",             "BODY_RIGHT_COLOR",   "color",  0, 0, 0),

    ("UI COLORS",              None,                "section", 0, 0, 0),
    ("Background Color",       "BACKGROUND_COLOR",   "color",  0, 0, 0),
    ("Info Text",              "TEXT_INFO_COLOR",    "color",  0, 0, 0),
    ("Label Text",             "LABEL_INFO_COLOR",   "color",  0, 0, 0),
]

# ------- SETTINGS STATE -------
scroll_offset = 0
hovered_row = -1
window_created = False
dragging_scrollbar = False

# ------- COLOR WHEEL STATE -------
color_picker_open = -1
WHEEL_RADIUS = 90
WHEEL_CENTER_HOLE = 20  # Small hole in center
BRIGHT_BAR_W = 20
BRIGHT_BAR_H = WHEEL_RADIUS * 2

# Current picker values
picker_hue = 0
picker_sat = 255
picker_val = 255
picker_dragging = None  # "wheel", "brightness", or None

# Pre-generated wheel (generated once)
_wheel_cache = None
_wheel_mask_cache = None


# ============================================================
#                    COLOR WHEEL HELPERS
# ============================================================

def _generate_wheel():
    """Generate HSV wheel image once, cache it."""
    global _wheel_cache, _wheel_mask_cache

    size = WHEEL_RADIUS * 2
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = WHEEL_RADIUS, WHEEL_RADIUS

    # Use numpy meshgrid for speed
    ys, xs = np.mgrid[0:size, 0:size]
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Ring mask
    ring = (dist >= WHEEL_CENTER_HOLE) & (dist <= WHEEL_RADIUS)

    # Hue from angle
    angle = np.arctan2(dy, dx)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)

    # Saturation from distance
    sat_ratio = np.clip((dist - WHEEL_CENTER_HOLE) / (WHEEL_RADIUS - WHEEL_CENTER_HOLE), 0, 1)
    sat = (sat_ratio * 255).astype(np.uint8)

    wheel[ring, 0] = hue[ring]
    wheel[ring, 1] = sat[ring]
    wheel[ring, 2] = 255
    mask[ring] = 255

    _wheel_cache = cv2.cvtColor(wheel, cv2.COLOR_HSV2BGR)
    _wheel_mask_cache = mask


def _get_wheel():
    """Get cached wheel, generate if needed."""
    if _wheel_cache is None:
        _generate_wheel()
    return _wheel_cache, _wheel_mask_cache


def _hsv_to_bgr(h, s, v):
    """Convert HSV to BGR tuple."""
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2]))


def _bgr_to_hsv(bgr):
    """Convert BGR tuple to HSV tuple."""
    bgr_arr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    return (int(hsv[0][0][0]), int(hsv[0][0][1]), int(hsv[0][0][2]))


def _get_popup_layout():
    """Calculate all popup positions."""
    popup_w = WHEEL_RADIUS * 2 + BRIGHT_BAR_W + 80
    popup_h = WHEEL_RADIUS * 2 + 120  # title + preview + buttons

    popup_x = (WINDOW_WIDTH - popup_w) // 2
    popup_y = (VISIBLE_HEIGHT - popup_h) // 2

    wheel_cx = popup_x + 25 + WHEEL_RADIUS
    wheel_cy = popup_y + 45 + WHEEL_RADIUS

    bar_x = wheel_cx + WHEEL_RADIUS + 20
    bar_y = wheel_cy - WHEEL_RADIUS

    preview_y = wheel_cy + WHEEL_RADIUS + 10
    btn_y = preview_y + 35

    return {
        "popup_x": popup_x, "popup_y": popup_y,
        "popup_w": popup_w, "popup_h": popup_h,
        "wheel_cx": wheel_cx, "wheel_cy": wheel_cy,
        "bar_x": bar_x, "bar_y": bar_y,
        "preview_y": preview_y, "btn_y": btn_y,
    }


# ============================================================
#                    DRAW COLOR PICKER
# ============================================================

def draw_color_picker(img):
    """Draw color wheel popup on the image."""
    global picker_hue, picker_sat, picker_val

    row_i = color_picker_open
    setting_name = SETTINGS_LAYOUT[row_i][1]
    current_bgr = getattr(S, setting_name)
    label = SETTINGS_LAYOUT[row_i][0]

    L = _get_popup_layout()
    wheel_bgr, wheel_mask = _get_wheel()

    # ------- DARK OVERLAY -------
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, VISIBLE_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # ------- POPUP BOX -------
    cv2.rectangle(img,
                 (L["popup_x"], L["popup_y"]),
                 (L["popup_x"] + L["popup_w"], L["popup_y"] + L["popup_h"]),
                 (40, 40, 40), -1)
    cv2.rectangle(img,
                 (L["popup_x"], L["popup_y"]),
                 (L["popup_x"] + L["popup_w"], L["popup_y"] + L["popup_h"]),
                 ACCENT, 2)

    # Title
    cv2.putText(img, f"Color: {label}",
               (L["popup_x"] + 15, L["popup_y"] + 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)

    # ------- COLOR WHEEL -------
    # Apply brightness to wheel
    wheel_display = wheel_bgr.copy()
    wheel_display = (wheel_display.astype(np.float32) * (picker_val / 255.0)).astype(np.uint8)

    # Draw wheel onto image
    wx = L["wheel_cx"] - WHEEL_RADIUS
    wy = L["wheel_cy"] - WHEEL_RADIUS
    size = WHEEL_RADIUS * 2

    # Bounds check
    if wy >= 0 and wx >= 0 and wy + size <= VISIBLE_HEIGHT and wx + size <= WINDOW_WIDTH:
        mask_bool = wheel_mask > 0
        roi = img[wy:wy + size, wx:wx + size]
        roi[mask_bool] = wheel_display[mask_bool]

    # Wheel border circle
    cv2.circle(img, (L["wheel_cx"], L["wheel_cy"]), WHEEL_RADIUS,
              BORDER, 1, cv2.LINE_AA)
    cv2.circle(img, (L["wheel_cx"], L["wheel_cy"]), WHEEL_CENTER_HOLE,
              BORDER, 1, cv2.LINE_AA)

    # ------- SELECTION INDICATOR ON WHEEL -------
    angle = (picker_hue / 179.0) * 2 * math.pi - math.pi
    sat_ratio = picker_sat / 255.0
    sel_dist = WHEEL_CENTER_HOLE + sat_ratio * (WHEEL_RADIUS - WHEEL_CENTER_HOLE)
    sel_x = int(L["wheel_cx"] + sel_dist * math.cos(angle))
    sel_y = int(L["wheel_cy"] + sel_dist * math.sin(angle))

    cv2.circle(img, (sel_x, sel_y), 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (sel_x, sel_y), 5, (0, 0, 0), 1, cv2.LINE_AA)

    # ------- BRIGHTNESS BAR -------
    bar_x = L["bar_x"]
    bar_y = L["bar_y"]

    for y_off in range(BRIGHT_BAR_H):
        val = 255 - int(y_off / BRIGHT_BAR_H * 255)
        color = _hsv_to_bgr(picker_hue, picker_sat, val)
        cv2.line(img,
                (bar_x, bar_y + y_off),
                (bar_x + BRIGHT_BAR_W, bar_y + y_off),
                color, 1)

    # Bar border
    cv2.rectangle(img, (bar_x, bar_y),
                 (bar_x + BRIGHT_BAR_W, bar_y + BRIGHT_BAR_H),
                 BORDER, 1)

    # Brightness indicator
    bright_y = bar_y + int((1 - picker_val / 255.0) * BRIGHT_BAR_H)
    bright_y = max(bar_y, min(bar_y + BRIGHT_BAR_H, bright_y))
    cv2.line(img, (bar_x - 3, bright_y), (bar_x + BRIGHT_BAR_W + 3, bright_y),
            (255, 255, 255), 2)

    # Brightness label
    cv2.putText(img, "Bright",
               (bar_x - 2, bar_y - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1)

    # ------- PREVIEW -------
    preview_y = L["preview_y"]
    selected_color = _hsv_to_bgr(picker_hue, picker_sat, picker_val)

    # Current color label
    cv2.putText(img, "Current:",
               (L["popup_x"] + 15, preview_y + 18),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    cv2.rectangle(img,
                 (L["popup_x"] + 80, preview_y + 3),
                 (L["popup_x"] + 130, preview_y + 25),
                 current_bgr, -1)
    cv2.rectangle(img,
                 (L["popup_x"] + 80, preview_y + 3),
                 (L["popup_x"] + 130, preview_y + 25),
                 BORDER, 1)

    # New color label
    cv2.putText(img, "New:",
               (L["popup_x"] + 140, preview_y + 18),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    cv2.rectangle(img,
                 (L["popup_x"] + 180, preview_y + 3),
                 (L["popup_x"] + 230, preview_y + 25),
                 selected_color, -1)
    cv2.rectangle(img,
                 (L["popup_x"] + 180, preview_y + 3),
                 (L["popup_x"] + 230, preview_y + 25),
                 BORDER, 1)

    # RGB values
    b, g, r = selected_color
    cv2.putText(img, f"R:{r} G:{g} B:{b}",
               (L["popup_x"] + 15, preview_y + 42),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # ------- BUTTONS -------
    btn_y = L["btn_y"]
    btn_w = 70
    btn_h = 28
    gap = 20

    # Apply button
    apply_x = L["popup_x"] + L["popup_w"] // 2 - btn_w - gap // 2
    cv2.rectangle(img, (apply_x, btn_y), (apply_x + btn_w, btn_y + btn_h),
                 VALUE_ON, -1)
    cv2.rectangle(img, (apply_x, btn_y), (apply_x + btn_w, btn_y + btn_h),
                 BORDER, 1)
    cv2.putText(img, "Apply",
               (apply_x + 12, btn_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Cancel button
    cancel_x = apply_x + btn_w + gap
    cv2.rectangle(img, (cancel_x, btn_y), (cancel_x + btn_w, btn_y + btn_h),
                 VALUE_OFF, -1)
    cv2.rectangle(img, (cancel_x, btn_y), (cancel_x + btn_w, btn_y + btn_h),
                 BORDER, 1)
    cv2.putText(img, "Cancel",
               (cancel_x + 8, btn_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ============================================================
#                    EVENT HANDLING
# ============================================================

def _handle_picker_events(event, x, y, flags):
    """Handle mouse events when color picker is open."""
    global color_picker_open, picker_hue, picker_sat, picker_val, picker_dragging

    L = _get_popup_layout()

    # ------- MOUSE UP: stop dragging -------
    if event == cv2.EVENT_LBUTTONUP:
        picker_dragging = None
        return

    # ------- DRAGGING -------
    if event == cv2.EVENT_MOUSEMOVE and picker_dragging:
        if picker_dragging == "wheel":
            dx = x - L["wheel_cx"]
            dy = y - L["wheel_cy"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0:
                angle = math.atan2(dy, dx)
                picker_hue = int((angle + math.pi) / (2 * math.pi) * 179)
                sat_ratio = (dist - WHEEL_CENTER_HOLE) / (WHEEL_RADIUS - WHEEL_CENTER_HOLE)
                picker_sat = int(max(0, min(255, sat_ratio * 255)))
            return

        elif picker_dragging == "brightness":
            bar_y = L["bar_y"]
            ratio = (y - bar_y) / BRIGHT_BAR_H
            picker_val = int(max(0, min(255, (1 - ratio) * 255)))
            return

    # ------- MOUSE DOWN -------
    if event == cv2.EVENT_LBUTTONDOWN:

        # Check wheel click
        dx = x - L["wheel_cx"]
        dy = y - L["wheel_cy"]
        dist = math.sqrt(dx * dx + dy * dy)

        if WHEEL_CENTER_HOLE <= dist <= WHEEL_RADIUS:
            picker_dragging = "wheel"
            angle = math.atan2(dy, dx)
            picker_hue = int((angle + math.pi) / (2 * math.pi) * 179)
            sat_ratio = (dist - WHEEL_CENTER_HOLE) / (WHEEL_RADIUS - WHEEL_CENTER_HOLE)
            picker_sat = int(max(0, min(255, sat_ratio * 255)))
            return

        # Check brightness bar
        bar_x = L["bar_x"]
        bar_y = L["bar_y"]
        if bar_x <= x <= bar_x + BRIGHT_BAR_W and bar_y <= y <= bar_y + BRIGHT_BAR_H:
            picker_dragging = "brightness"
            ratio = (y - bar_y) / BRIGHT_BAR_H
            picker_val = int(max(0, min(255, (1 - ratio) * 255)))
            return

        # Check Apply button
        btn_y = L["btn_y"]
        btn_w = 70
        btn_h = 28
        gap = 20
        apply_x = L["popup_x"] + L["popup_w"] // 2 - btn_w - gap // 2

        if apply_x <= x <= apply_x + btn_w and btn_y <= y <= btn_y + btn_h:
            setting_name = SETTINGS_LAYOUT[color_picker_open][1]
            new_color = _hsv_to_bgr(picker_hue, picker_sat, picker_val)
            setattr(S, setting_name, new_color)
            print(f"  {setting_name} = {new_color}")
            color_picker_open = -1
            picker_dragging = None
            return

        # Check Cancel button
        cancel_x = apply_x + btn_w + gap
        if cancel_x <= x <= cancel_x + btn_w and btn_y <= y <= btn_y + btn_h:
            color_picker_open = -1
            picker_dragging = None
            return

        # Click outside popup = cancel
        if not (L["popup_x"] <= x <= L["popup_x"] + L["popup_w"] and
                L["popup_y"] <= y <= L["popup_y"] + L["popup_h"]):
            color_picker_open = -1
            picker_dragging = None


def handle_events(event, x, y, flags, param):
    global hovered_row, scroll_offset, dragging_scrollbar
    global color_picker_open, picker_hue, picker_sat, picker_val

    max_scroll = get_max_scroll()
    content_width = WINDOW_WIDTH - SCROLLBAR_WIDTH - 5

    # ------- COLOR PICKER IS OPEN -------
    if color_picker_open >= 0:
        _handle_picker_events(event, x, y, flags)
        return

    # ------- SCROLL WHEEL -------
    if event == cv2.EVENT_MOUSEWHEEL:
        scroll_speed = ROW_HEIGHT
        if flags > 0:
            scroll_offset = max(0, scroll_offset - scroll_speed)
        else:
            scroll_offset = min(max_scroll, scroll_offset + scroll_speed)
        return

    # ------- SCROLLBAR DRAG -------
    if event == cv2.EVENT_LBUTTONDOWN:
        sb_x = WINDOW_WIDTH - SCROLLBAR_WIDTH
        if x >= sb_x and y >= HEADER_HEIGHT:
            dragging_scrollbar = True
            sb_track_h = VISIBLE_HEIGHT - HEADER_HEIGHT
            ratio = (y - HEADER_HEIGHT) / sb_track_h
            scroll_offset = int(ratio * max_scroll)
            scroll_offset = max(0, min(max_scroll, scroll_offset))
            return

    if event == cv2.EVENT_MOUSEMOVE and dragging_scrollbar:
        sb_track_h = VISIBLE_HEIGHT - HEADER_HEIGHT
        ratio = (y - HEADER_HEIGHT) / sb_track_h
        scroll_offset = int(ratio * max_scroll)
        scroll_offset = max(0, min(max_scroll, scroll_offset))
        return

    if event == cv2.EVENT_LBUTTONUP:
        dragging_scrollbar = False

    # ------- ROW HOVER -------
    if event == cv2.EVENT_MOUSEMOVE:
        hovered_row = -1
        if x < content_width:
            for i in range(len(SETTINGS_LAYOUT)):
                row_y = get_row_y(i)
                if row_y < HEADER_HEIGHT:
                    continue
                if row_y <= y <= row_y + ROW_HEIGHT:
                    if SETTINGS_LAYOUT[i][2] != "section":
                        hovered_row = i
                    break

    # ------- ROW CLICK -------
    elif event == cv2.EVENT_LBUTTONDOWN:
        if x >= WINDOW_WIDTH - SCROLLBAR_WIDTH:
            return

        for i, (label, setting_name, stype, smin, smax, step) in enumerate(SETTINGS_LAYOUT):
            row_y = get_row_y(i)

            if row_y < HEADER_HEIGHT:
                continue
            if not (row_y <= y <= row_y + ROW_HEIGHT):
                continue

            if stype == "bool":
                sw_x = content_width - 75
                sw_y = row_y + 10
                if sw_x <= x <= sw_x + 55 and sw_y <= y <= sw_y + 22:
                    current = getattr(S, setting_name)
                    setattr(S, setting_name, not current)
                    print(f"  {setting_name} = {not current}")

            elif stype == "int":
                current = getattr(S, setting_name)
                btn_y = row_y + 9
                minus_x = content_width - 140
                plus_x = minus_x + 80

                if minus_x <= x <= minus_x + 30 and btn_y <= y <= btn_y + 24:
                    new_val = max(smin, current - step)
                    setattr(S, setting_name, new_val)
                    print(f"  {setting_name} = {new_val}")
                elif plus_x <= x <= plus_x + 30 and btn_y <= y <= btn_y + 24:
                    new_val = min(smax, current + step)
                    setattr(S, setting_name, new_val)
                    print(f"  {setting_name} = {new_val}")

            elif stype == "color":
                swatch_x = content_width - 100
                swatch_y = row_y + 10
                if swatch_x <= x <= swatch_x + 80 and swatch_y <= y <= swatch_y + 22:
                    # Open picker and set initial HSV from current color
                    color_picker_open = i
                    current = getattr(S, setting_name)
                    picker_hue, picker_sat, picker_val = _bgr_to_hsv(current)


# ============================================================
#              SETTINGS ROWS (unchanged logic)
# ============================================================

def get_total_content_height():
    return HEADER_HEIGHT + PADDING + len(SETTINGS_LAYOUT) * ROW_HEIGHT + PADDING

def get_max_scroll():
    total = get_total_content_height()
    return max(0, total - VISIBLE_HEIGHT)

def get_row_y(i):
    return HEADER_HEIGHT + PADDING + i * ROW_HEIGHT - scroll_offset


def create_settings_image():
    img = np.full((VISIBLE_HEIGHT, WINDOW_WIDTH, 3), BG_COLOR, dtype=np.uint8)

    # Header
    cv2.rectangle(img, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), SECTION_COLOR, -1)
    cv2.putText(img, "SETTINGS",
               (WINDOW_WIDTH // 2 - 55, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT, 2)
    cv2.putText(img, "Scroll: Mouse Wheel",
               (WINDOW_WIDTH // 2 - 75, 43),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.line(img, (0, HEADER_HEIGHT), (WINDOW_WIDTH, HEADER_HEIGHT), BORDER, 1)

    content_width = WINDOW_WIDTH - SCROLLBAR_WIDTH - 5

    for i, (label, setting_name, stype, smin, smax, step) in enumerate(SETTINGS_LAYOUT):
        row_y = get_row_y(i)

        if row_y + ROW_HEIGHT < HEADER_HEIGHT or row_y > VISIBLE_HEIGHT:
            continue
        if row_y < HEADER_HEIGHT:
            continue

        if stype == "section":
            cv2.rectangle(img, (0, row_y), (content_width, row_y + ROW_HEIGHT),
                         SECTION_COLOR, -1)
            cv2.line(img, (15, row_y + ROW_HEIGHT - 2),
                    (content_width - 15, row_y + ROW_HEIGHT - 2), ACCENT, 1)
            cv2.putText(img, label, (15, row_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)

        elif stype == "bool":
            value = getattr(S, setting_name)
            if i == hovered_row:
                cv2.rectangle(img, (0, row_y), (content_width, row_y + ROW_HEIGHT),
                             (50, 50, 50), -1)
            cv2.putText(img, label, (20, row_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

            sw_x = content_width - 75
            sw_y = row_y + 10
            sw_w, sw_h = 55, 22

            if value:
                cv2.rectangle(img, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h),
                             VALUE_ON, -1, cv2.LINE_AA)
                cv2.circle(img, (sw_x+sw_w-11, sw_y+11), 9,
                          (255,255,255), -1, cv2.LINE_AA)
                cv2.putText(img, "ON", (sw_x+5, sw_y+16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            else:
                cv2.rectangle(img, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h),
                             VALUE_OFF, -1, cv2.LINE_AA)
                cv2.circle(img, (sw_x+11, sw_y+11), 9,
                          (200,200,200), -1, cv2.LINE_AA)
                cv2.putText(img, "OFF", (sw_x+25, sw_y+16),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            cv2.rectangle(img, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h),
                         BORDER, 1, cv2.LINE_AA)

        elif stype == "int":
            value = getattr(S, setting_name)
            if i == hovered_row:
                cv2.rectangle(img, (0, row_y), (content_width, row_y + ROW_HEIGHT),
                             (50, 50, 50), -1)
            cv2.putText(img, label, (20, row_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

            btn_y = row_y + 9
            btn_h = 24
            minus_x = content_width - 140

            cv2.rectangle(img, (minus_x, btn_y), (minus_x+30, btn_y+btn_h),
                         BTN_HOVER if (i==hovered_row) else BTN_COLOR, -1)
            cv2.rectangle(img, (minus_x, btn_y), (minus_x+30, btn_y+btn_h), BORDER, 1)
            cv2.putText(img, "-", (minus_x+10, btn_y+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

            val_text = str(value)
            ts = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            val_x = minus_x + 35 + (40 - ts[0]) // 2
            cv2.putText(img, val_text, (val_x, btn_y+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)

            plus_x = minus_x + 80
            cv2.rectangle(img, (plus_x, btn_y), (plus_x+30, btn_y+btn_h),
                         BTN_HOVER if (i==hovered_row) else BTN_COLOR, -1)
            cv2.rectangle(img, (plus_x, btn_y), (plus_x+30, btn_y+btn_h), BORDER, 1)
            cv2.putText(img, "+", (plus_x+8, btn_y+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        elif stype == "color":
            value = getattr(S, setting_name)
            if i == hovered_row:
                cv2.rectangle(img, (0, row_y), (content_width, row_y + ROW_HEIGHT),
                             (50, 50, 50), -1)
            cv2.putText(img, label, (20, row_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

            swatch_x = content_width - 100
            swatch_y = row_y + 10
            cv2.rectangle(img, (swatch_x, swatch_y),
                         (swatch_x+80, swatch_y+22), value, -1)
            cv2.rectangle(img, (swatch_x, swatch_y),
                         (swatch_x+80, swatch_y+22), BORDER, 1)
            cv2.putText(img, "click",
                       (swatch_x + 25, swatch_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                       (0,0,0) if sum(value) > 384 else (255,255,255), 1)

        if stype != "section":
            cv2.line(img, (20, row_y + ROW_HEIGHT - 1),
                    (content_width - 20, row_y + ROW_HEIGHT - 1), (45,45,45), 1)

    # Scrollbar
    max_scroll = get_max_scroll()
    if max_scroll > 0:
        sb_x = WINDOW_WIDTH - SCROLLBAR_WIDTH
        sb_track_y = HEADER_HEIGHT
        sb_track_h = VISIBLE_HEIGHT - HEADER_HEIGHT
        cv2.rectangle(img, (sb_x, sb_track_y), (WINDOW_WIDTH, VISIBLE_HEIGHT),
                     SCROLLBAR_BG, -1)
        visible_ratio = VISIBLE_HEIGHT / get_total_content_height()
        thumb_h = max(30, int(sb_track_h * visible_ratio))
        scroll_ratio = scroll_offset / max_scroll if max_scroll > 0 else 0
        thumb_y = sb_track_y + int((sb_track_h - thumb_h) * scroll_ratio)
        cv2.rectangle(img, (sb_x+2, thumb_y), (WINDOW_WIDTH-2, thumb_y+thumb_h),
                     SCROLLBAR_HOVER if dragging_scrollbar else SCROLLBAR_FG,
                     -1, cv2.LINE_AA)

    # Color picker overlay
    if color_picker_open >= 0:
        draw_color_picker(img)

    return img


# ============================================================
#                    WINDOW MANAGEMENT
# ============================================================

def open_window():
    global window_created
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, handle_events)
    window_created = True

def close_window():
    global window_created
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except:
        pass
    window_created = False

def update():
    global window_created
    if S.MENU_OPEN:
        if not window_created:
            open_window()
        img = create_settings_image()
        cv2.imshow(WINDOW_NAME, img)
    else:
        if window_created:
            close_window()