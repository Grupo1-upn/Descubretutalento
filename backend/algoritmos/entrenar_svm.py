import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

# Cargar el dataset
data = pd.read_csv('C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\data\\test_data3.0.csv', delimiter=';')

# Separar características (X) y etiqueta (y)
X = data.drop('Orientacion_Vocacional', axis=1)
y = data['Orientacion_Vocacional']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo SVM con cálculo de probabilidades
model = SVC(probability=True)

# Definir los hiperparámetros a probar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Configurar GridSearchCV con validación cruzada
grid = GridSearchCV(model, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Obtener el mejor modelo y predecir
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
y_pred_proba = best_svm.predict_proba(X_test)

# Verificar tamaños
print(f"Tamaño de y_test: {len(y_test)}")
print(f"Tamaño de y_pred_proba: {y_pred_proba.shape[0]}")

# ======== Calcular y Graficar la Curva ROC por Cada Clase ========
# Binarizar las etiquetas para ROC multiclase
y_test_bin = label_binarize(y_test, classes=np.unique(y))

# Inicializamos las variables para las curvas ROC
fpr = {}
tpr = {}
roc_auc = {}

# Calculamos la curva ROC y el AUC por cada clase
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficamos las curvas ROC
plt.figure(figsize=(10, 8))
for i in range(y_test_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Línea de referencia de azar
plt.plot([0, 1], [0, 1], 'k--', label='Azar')

# Configuración de la gráfica
plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

# ======== Guardar el Modelo Entrenado ========
joblib.dump(best_svm, 'C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\modelos\\modelo_svm.pkl')

# Obtener el mejor modelo y predecir
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
y_pred_proba = best_svm.predict_proba(X_test)

# Calcular la precisión del modelo optimizado
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Model Accuracy: {accuracy * 100:.2f}%')


# ======== Mostrar la Matriz de Confusión ========
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Mostrar la matriz de confusión para el mejor modelo SVM
plot_confusion_matrix(y_test, y_pred, "SVM")

# ======== Guardar el Modelo Entrenado ========
joblib.dump(best_svm, 'C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\modelos\\modelo_svm.pkl')

# ======== Calcular otras métricas ========
accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Imprimir las métricas
print(f'SVM Model Accuracy: {accuracy * 100:.2f}%')
print(f'Weighted Precision: {precision_weighted * 100:.2f}%')
print(f'Weighted F1 Score: {f1 * 100:.2f}%')

# Generar datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Validación cruzada para obtener las métricas en entrenamiento y validación
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for train_index, val_index in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Verificar si ambas clases están presentes
    if len(set(y_train_fold)) > 1 and len(set(y_val_fold)) > 1:
        # Entrenamiento
        svm_model.fit(X_train_fold, y_train_fold)
        
        # Predicciones
        y_train_pred = svm_model.predict(X_train_fold)
        y_train_prob = svm_model.predict_proba(X_train_fold)[:, 1]
        y_val_pred = svm_model.predict(X_val_fold)
        y_val_prob = svm_model.predict_proba(X_val_fold)[:, 1]
        
        # Calcular precisión y pérdida para entrenamiento y validación
        train_accuracies.append(accuracy_score(y_train_fold, y_train_pred))
        val_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
        train_losses.append(log_loss(y_train_fold, y_train_prob, labels=[0, 1]))  # Especificamos las etiquetas
        val_losses.append(log_loss(y_val_fold, y_val_prob, labels=[0, 1]))  # Especificamos las etiquetas

# Graficar la precisión y la pérdida
plt.figure(figsize=(12, 6))

# Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Fold')
plt.ylabel('Precisión')
plt.title('Training and Validation Accuracy (SVM)')
plt.legend()

# Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Fold')
plt.ylabel('Log Loss')
plt.title('Loss during training and validation (SVM)')
plt.legend()

plt.tight_layout()
plt.show()
