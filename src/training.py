import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

def entrenar_modelo(nombre_modelo='model_rock_paper_scissors.h5'):
    print("loading dataset...")
    (ds_train, ds_val), ds_info = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    IMG_SIZE = 224
    BATCH_SIZE = 32

    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds_train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
    ds_val = ds_val.map(preprocess).batch(BATCH_SIZE)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Entrenando modelo...")
    model.fit(ds_train, validation_data=ds_val, epochs=5)
    model.save(nombre_modelo)
    print("Entrenamiento terminado")
