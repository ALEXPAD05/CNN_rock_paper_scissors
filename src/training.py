import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models # type: ignore

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

    # Preprocesamiento base
    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Aumentación de datos
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.05)

        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # 0, 1, 2, 3
        image = tf.image.rot90(image, k)

        image = tf.image.random_crop(tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 10, IMG_SIZE + 10), size=[IMG_SIZE, IMG_SIZE, 3])
        return image, label

    ds_train = ds_train.map(preprocess).map(augment).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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