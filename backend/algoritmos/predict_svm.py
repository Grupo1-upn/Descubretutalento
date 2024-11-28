import sys
import joblib
import json
import numpy as np

# Cargar el modelo SVM entrenado
model_path = 'C:/Users/gsucf/Desktop/DescubreTuTalento/machine-learning/modelos/modelo_svm.pkl'
modelo_svm = joblib.load(model_path)

# Obtener los datos pasados como argumento desde Node.js
input_data = sys.argv[1]

# Convertir el argumento de cadena de texto a un diccionario (intereses y aptitudes)
respuestas = json.loads(input_data)

# Combinar intereses y aptitudes en una sola lista de características
input_features = np.array(respuestas['intereses'] + respuestas['aptitudes']).reshape(1, -1)

# Hacer la predicción con el modelo SVM
prediccion = modelo_svm.predict(input_features)

# Imprimir el resultado para que Node.js lo capture
print(prediccion[0])  # Esto es lo que se devuelve a Node.js
