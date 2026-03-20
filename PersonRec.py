import cv2
import numpy as np
import mediapipe as mp
import drawFace as dF
import drawHand as dH
import drawBody as dB         
import Settings as S
import SettingsMenu as SM
import time

# ------- CAMERA -------
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, S.FRAME_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, S.FRAME_HEIGHT)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WINDOW_NAME = "Face & Hand Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, S.FRAME_WIDTH, S.FRAME_HEIGHT)
aspect_ratio = S.FRAME_WIDTH / S.FRAME_HEIGHT
last_window_w = S.FRAME_WIDTH
last_window_h = S.FRAME_HEIGHT

with dF.FaceLandmarker.create_from_options(dF.options) as face_landmarker, \
     dH.HandLandmarker.create_from_options(dH.options) as hand_landmarker, \
     dB.PoseLandmarker.create_from_options(dB.options) as body_landmarker:

    frame_timestamp = 0
    frame_count = 0

    prev_face_result = None
    prev_hand_result = None
    prev_body_result = None    # <-- NEW

    fps = 0
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ------- DETECTION -------
        if frame_count % S.PROCESS_EVERY_N == 0:
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if S.DETECT_FACE:
                prev_face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp)
            else:
                prev_face_result = None

            if S.DETECT_HANDS:
                prev_hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
            else:
                prev_hand_result = None

            if S.DETECT_BODY:                                               # <-- NEW
                prev_body_result = body_landmarker.detect_for_video(mp_image, frame_timestamp)
            else:
                prev_body_result = None

        frame_timestamp += 33
        frame_count += 1

        # ------- DISPLAY FRAME -------
        if S.SHOW_VIDEO:
            display = frame
        else:
            display = np.full_like(frame, S.BACKGROUND_COLOR, dtype=np.uint8)

        # ------- DRAWING -------
        # Draw body FIRST so face and hands draw on top
        if S.DRAW_BODY and prev_body_result:                                # <-- NEW
            display = dB.draw_body(display, prev_body_result)

        if S.DRAW_FACE and prev_face_result:
            if S.FACE_OPTIMIZED:
                display = dF.draw_face_optimized(display, prev_face_result)
            else:
                display = dF.draw_face(display, prev_face_result)

        if S.DRAW_HANDS and prev_hand_result:
            display = dH.draw_hands(display, prev_hand_result)

        # ------- INFO TEXT -------
        y_offset = 30

        if S.SHOW_FPS:
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 0.5:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            cv2.putText(display, f"FPS: {int(fps)}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, S.TEXT_INFO_COLOR, 1)
            y_offset += 30

        if S.SHOW_FACE_COUNT and prev_face_result and prev_face_result.face_landmarks:
            cv2.putText(display, f"Faces: {len(prev_face_result.face_landmarks)}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, S.TEXT_INFO_COLOR, 2)
            y_offset += 30

        if S.SHOW_BODY_TEXT and prev_body_result and prev_body_result.pose_landmarks:  # <-- NEW
            body_info = dB.get_body_info(prev_body_result)
            for info in body_info:
                cv2.putText(display, info,
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           S.TEXT_INFO_COLOR, 2)
                y_offset += 30

        if S.SHOW_FINGER_COUNT and prev_hand_result and prev_hand_result.hand_landmarks:
            cv2.putText(display, f"Hands: {len(prev_hand_result.hand_landmarks)}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, S.TEXT_INFO_COLOR, 2)
            y_offset += 30
            for i, hand_lm in enumerate(prev_hand_result.hand_landmarks):
                mp_name = prev_hand_result.handedness[i][0].category_name
                display_name = "Left" if mp_name == "Right" else "Right"
                fingers = dH.count_fingers(hand_lm, mp_name)
                cv2.putText(display, f"{display_name}: {fingers} fingers",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           S.TEXT_INFO_COLOR, 2)
                y_offset += 30

        menu_text = "M: Close Settings" if S.MENU_OPEN else "M: Open Settings"
        cv2.putText(display, menu_text,
                   (display.shape[1] - 200, display.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ------- ASPECT RATIO LOCK -------
        try:
            rect = cv2.getWindowImageRect(WINDOW_NAME)
            win_w, win_h = rect[2], rect[3]
            if win_w > 0 and win_h > 0:
                if win_w != last_window_w or win_h != last_window_h:
                    w_change = abs(win_w - last_window_w)
                    h_change = abs(win_h - last_window_h)
                    if w_change >= h_change:
                        new_w = win_w
                        new_h = int(win_w / aspect_ratio)
                    else:
                        new_h = win_h
                        new_w = int(win_h * aspect_ratio)
                    cv2.resizeWindow(WINDOW_NAME, new_w, new_h)
                    last_window_w = new_w
                    last_window_h = new_h
        except:
            pass

        cv2.imshow(WINDOW_NAME, display)
        SM.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('o'):
            S.MENU_OPEN = not S.MENU_OPEN

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

video.release()
SM.close_window()
cv2.destroyAllWindows()