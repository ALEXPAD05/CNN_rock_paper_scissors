import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def cam(nombre_modelo='model_rock_paper_scissors.h5'):
    model = load_model(nombre_modelo, compile=False)
    clases = ['piedra', 'papel', 'tijera']
    IMG_SIZE = 224

    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y = 100, 100
        roi = frame[y:y+IMG_SIZE, x:x+IMG_SIZE]
        input_img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        input_img = input_img.astype('float32') / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        pred = model.predict(input_img)
        class_id = np.argmax(pred)
        confidence = pred[0][class_id]
        label = f"{clases[class_id]} "

        cv2.rectangle(frame, (x, y), (x+IMG_SIZE, y+IMG_SIZE), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Piedra, Papel o Tijera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
