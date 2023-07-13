import cv2
import csv
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around the y-axis for correct handedness output.
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam/video input:
cap = cv2.VideoCapture("thanks.mp4")
cap2 = cv2.VideoCapture("water.mp4")

vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("output_video/thank.mp4", vid_cod, 20.0, (640, 480))

FILE_NAME = 'coords.csv'
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.5,
        static_image_mode=True) as hands:
    while cap.isOpened():
        success, image = cap.read()
        # success, image2 = cap2.read()
        # ret, out_image = cap2.read()
        if not success or image is None:
            print("Ignoring empty camera frame.")
            break

        success2, image2 = cap2.read()
        if not success2 or image2 is None:
            print("Ignoring empty camera frame.")
            break

        ret, out_image = cap2.read()
        if not ret or out_image is None:
            print("Ignoring empty camera frame.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if image.shape[2] != 3:
            continue  # Skip frames with unexpected number of color channels

        image_height, image_width, _ = image.shape

        image2.flags.writeable = False
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        results2 = hands.process(image2)
        if image2.shape[2] != 3:
            continue  # Skip frames with unexpected number of color channels

        image_height2, image_width2, _ = image2.shape

        image.flags.writeable = True
        image2.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        with open(FILE_NAME, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        coordinates = [landmrk.x, landmrk.y, landmrk.z]
                        writer.writerow(coordinates)
                        cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(
                        out_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,

                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if results2 is not None and results2.multi_hand_landmarks:
                for hand_landmarks in results2.multi_hand_landmarks:
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        coordinates = [landmrk.x, landmrk.y, landmrk.z]
                        writer.writerow(coordinates)
                        cx, cy = landmrk.x * image_width2, landmrk.y * image_height2
                        cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3)
                    # connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)


                    mp_drawing.draw_landmarks(
                        out_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('video', out_image)

        #cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cap2.release()
output.release()
