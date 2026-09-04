import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
frame_count = 0
ANALYZE_EVERY_N_FRAMES = 10
last_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
        try:
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False
            )

            last_faces = [r for r in results if r['face_confidence'] > 0.5]

            for face in last_faces:
                print(
                    f"Age: {face['age']} | "
                    f"Gender: {face['dominant_gender']} | "
                    f"Emotion: {face['dominant_emotion']} | "
                    f"Race: {face['dominant_race']} | "
                    f"Face Confidence: {face['face_confidence']:.2f}"
                )
        except Exception as e:
            print("Err:", e)

    for face in last_faces:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{face['dominant_gender']}, {face['age']} lat, {face['dominant_emotion']}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()