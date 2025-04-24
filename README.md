# Proyecto: Clasificación de Piedra, Papel o Tijera con CNN

Este es un proyecto de final de cuatrimestre el cual es un entrenamiento de una dataset de tensorflow el cual se encarga de identificar que ademan estamos haciendo con nuestra mano en el caso de ser piedra, papel o tijera.

Este proyecto utiliza una Red Neuronal Convolucional (CNN) para clasificar imágenes de "piedra", "papel" y "tijera". El flujo de trabajo incluye entrenamiento del modelo, predicción en tiempo real con la cámara y pruebas con imágenes preexistentes.

## Archivos del Proyecto

1. **main.py**: Coordina la ejecución del proyecto.
2. **rock_paper_scissors.py**: Predicción en tiempo real utilizando la cámara.
3. **rps_test.py**: Pruebas del modelo con imágenes de la carpeta `test`.
4. **training.py**: Entrenamiento del modelo CNN.

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