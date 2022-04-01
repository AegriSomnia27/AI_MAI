import cv2
import mediapipe as mp
import numpy as np

def is_sign_of_the_horns(hand_landmarks) -> bool:
    sign_of_horns_status = { "INDEX": True, "MIDDLE": False, "RING": False,
                             "PINKY": True, "THUMB": True }

    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                       mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_status = { "INDEX": False, "MIDDLE": False, "RING": False,
                        "PINKY": False, "THUMB": False}
    
    # check if index, middle, ring and pinky fingers are up
    for tip_index in fingers_tips_ids:
        finger_name = tip_index.name.split("_")[0]

        if(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
            fingers_status[finger_name] = True
    
    # check if the thumb in the right position
    thumb_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    thumb_finger_tip_position = np.array([thumb_finger_tip.x, thumb_finger_tip.y, thumb_finger_tip.z])
    middle_finger_pip_position = np.array([middle_finger_pip.x, middle_finger_pip.y, middle_finger_pip.z])

    distance = np.linalg.norm(thumb_finger_tip_position - middle_finger_pip_position)
    # print('calculated distance: ', distance)
    confidence = 0.2
    
    if distance < confidence:
        fingers_status["THUMB"] = True
    
    # return bool if it's a horns
    # print(fingers_status)
    # print(sign_of_horns_status == fingers_status)
    return (sign_of_horns_status == fingers_status)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        horns = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                if(is_sign_of_the_horns(hand_landmarks)):
                    horns.append("HAIL, SATAN")
        # print(horns)
        flipped_image = cv2.flip(image.copy(), 1)

        if horns:
            cv2.putText(flipped_image, ' '.join(horns), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(flipped_image, ' '.join(horns), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA),

        cv2.imshow('MediaPipe Hands', flipped_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()  