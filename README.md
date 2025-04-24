# Proyecto: Clasificación de Piedra, Papel o Tijera con CNN

Este es un proyecto de final de cuatrimestre el cual es un entrenamiento de una dataset de tensorflow el cual se encarga de identificar que ademan estamos haciendo con nuestra mano en el caso de ser piedra, papel o tijera. todo esto con una red neuronal convolucional.

## Archivos del Proyecto

1. **main.py**: Coordina la ejecución del proyecto.
2. **rock_paper_scissors.py**: Predicción en tiempo real utilizando la cámara.
3. **rps_test.py**: Pruebas del modelo con imágenes de la carpeta `test`.
4. **training.py**: Entrenamiento del modelo CNN.

## Orden de carpetas
```
CNN/
├── src/                         # Carpeta con los archivos principales del proyecto
│   └── training.py                 # Entrenamiento del modelo CNN.4
│   └── rps_test.py                 # Verificacion de nuestra Red mediante imagenes
│   └── rock_paper_scissors.py      # Verificacion de nuestra Red mediante la camara
├── test/                        # Carpeta con imágenes de prueba para el modelo.
│   └── 1.png                       # Ejemplo de imagen de prueba.
│   └── 2.png                       # Ejemplo de imagen de prueba.
│   └── 3.png                       # Ejemplo de imagen de prueba.
├── venv/                        # Carpeta del entorno virtual (opcional)
├── .gitattributes               # Archivo para subir modelo
├── .gitignore                   # Archivo para subir seleccion de que subir a github
├── main.py                      # Archivo principal que coordina la ejecución del proyecto.
├── model_rock_paper_scissors.h5 # Modelo entrenado
├── README.md                    # Este archivo.
└── requirements.txt             # Archivo con las dependencias del proyecto.
```

## Requisitos

Instala las dependencias desde el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Uso
1. Entrenar el modelo
```bash
python main.py rps_train
```

2. Usar la cámara para predicciones:

```bash
python main.py rps_cam
```

3. Probar el modelo con imágenes de prueba:

```bash
python main.py rps_test
```