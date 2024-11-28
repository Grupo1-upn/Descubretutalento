import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


# Cargar el dataset
data = pd.read_csv('C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\data\\test_data3.0.csv', encoding='ISO-8859-1', sep=';')

# Mostrar las primeras filas del DataFrame
print(data.head())

# Limpiar nombres de columnas
data.columns = data.columns.str.strip().str.replace('ï»¿', '', regex=False)

# Verificar y convertir la columna de Orientación Vocacional
data['Orientacion_Vocacional'] = pd.factorize(data['Orientacion_Vocacional'])[0]  # Mapeo de carreras a valores numéricos
print("Valores únicos de Orientación Vocacional después del mapeo:", data['Orientacion_Vocacional'].unique())

# Eliminar filas con NaN en Orientación Vocacional (si hay)
data = data.dropna(subset=['Orientacion_Vocacional'])

# Comprobar el número de filas restantes
print(f"Número de muestras restantes: {data.shape[0]}")

# Separar las características (features) del objetivo (target)
X = data.drop('Orientacion_Vocacional', axis=1)
y = data['Orientacion_Vocacional']

# Verificar si hay datos en X y y
print(f"Número de características: {X.shape[0]}")
print(f"Número de etiquetas: {y.shape[0]}")

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ======== Construir el Modelo de Redes Neuronales (ANN) ========
model = Sequential()

# Capa de entrada
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Capa de entrada con 64 neuronas

# Capas ocultas
model.add(Dense(32, activation='relu'))  # Primera capa oculta con 32 neuronas
model.add(Dense(16, activation='relu'))  # Segunda capa oculta con 16 neuronas

# Capa de salida
model.add(Dense(len(y.unique()), activation='softmax'))  # Capa de salida para múltiples clases

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ======== Entrenar el Modelo ========
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# ======== Evaluar el Modelo ========
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Convertir probabilidades a clases

# Calcular la precisión
accuracy_ann = accuracy_score(y_test, y_pred_classes)
print(f'Precisión del modelo de Redes Neuronales (ANN): {accuracy_ann * 100:.2f}%')

# ======== Guardar el Modelo Entrenado ========
model.save('C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\modelos\\modelo_ann.h5')

# ======== Graficar la Matriz de Confusión ========
def plot_confusion_matrix(y_true, y_pred, title="Redes Neuronales ANN"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Mostrar la matriz de confusión
plot_confusion_matrix(y_test, y_pred_classes, title="Artificial Neural Networks (ANN)")

# ======== Calcular precisión, F1-score y exactitud ========
accuracy_ann = accuracy_score(y_test, y_pred_classes)
precision_ann = precision_score(y_test, y_pred_classes, average='weighted')
f1_ann = f1_score(y_test, y_pred_classes, average='weighted')

# Imprimir métricas
print(f'Accuracy of the Neural Network Model (ANN): {accuracy_ann * 100:.2f}%')
print(f'Weighted Precision: {precision_ann * 100:.2f}%')
print(f'Weighted F1 Score: {f1_ann * 100:.2f}%')


# ======== Entrenar el Modelo y Guardar el Historial ========
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(X_test, y_test))

# ======== Gráficos de Pérdida y Precisión ========
# Gráfico de la precisión en el entrenamiento y validación
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Gráfico de la pérdida en el entrenamiento y validación
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ======== Curva ROC para cada clase ========
# Convertir las etiquetas a formato binario para cada clase
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_pred_prob = model.predict(X_test)

# Crear la curva ROC para cada clase
n_classes = y_test_bin.shape[1]
fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal para referencia
plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='best')
plt.show()

# ======== Curva de Precisión-Recall ========
# Curva de precisión-recall para cada clase
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title(f'Precision-Recall Curve for Class {i}')
plt.show()