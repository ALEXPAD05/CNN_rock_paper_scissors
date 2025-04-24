import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def testing(carpeta='test', nombre_modelo='model_rock_paper_scissors.h5'):
    model = load_model(nombre_modelo, compile=False)
    clases = ['piedra', 'papel', 'tijera']
    IMG_SIZE = 224

    # Verifica que la carpeta existe
    if not os.path.exists(carpeta):
        print(f"La carpeta '{carpeta}' no existe.")
        return

    for nombre_archivo in os.listdir(carpeta):
        ruta_imagen = os.path.join(carpeta, nombre_archivo)

        if not (ruta_imagen.lower().endswith('.jpg') or ruta_imagen.lower().endswith('.png')):
            continue  # Saltar archivos que no son imágenes

        # Leer imagen y preprocesar
        img = cv2.imread(ruta_imagen)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        input_img = img_resized.astype('float32') / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Predicción
        pred = model.predict(input_img)
        class_id = np.argmax(pred)
        confidence = pred[0][class_id]

        # Mostrar resultado
        print(f"{nombre_archivo}: {clases[class_id]} ({confidence*100:.1f}%)")

        # Opcional: mostrar imagen con predicción
        cv2.putText(img_resized, f"{clases[class_id]} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Resultado", img_resized)
        cv2.waitKey(0)

    cv2.destroyAllWindows()