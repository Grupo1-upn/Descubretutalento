import sys
import joblib
import json
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('C:/Users/gsucf/Desktop/DescubreTuTalento/machine-learning/modelos/modelo_ann.h5')

# Obtener los datos de entrada pasados desde Node.js
input_data = sys.argv[1]

# Convertir el argumento de cadena de texto a un diccionario (intereses y aptitudes)
respuestas = json.loads(input_data)

# Combinar intereses y aptitudes en una sola lista de características
input_features = np.array(respuestas['intereses'] + respuestas['aptitudes']).reshape(1, -1)

# Hacer la predicción
prediccion = model.predict(input_features)
resultado = np.argmax(prediccion, axis=1)  # Obtener la clase con mayor probabilidad

# Imprimir el resultado
print(resultado[0])  # Este resultado se captura en Node.js
