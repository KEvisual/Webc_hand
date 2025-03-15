import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # Untuk menggambar landmark

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ubah frame ke RGB (bukan grayscale)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Deteksi jumlah jari yang terbuka
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP, 
                mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                mp_hands.HandLandmark.RING_FINGER_TIP, 
                mp_hands.HandLandmark.PINKY_TIP
            ]
            count = 0
            for tip in finger_tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    count += 1

            # Menampilkan jumlah jari yang terbuka
            cv2.putText(frame, f'Jari Terbuka: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pindahkan release dan destroyAllWindows ke luar loop
cap.release()
cv2.destroyAllWindows()
