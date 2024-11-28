import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
import joblib
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification

# Cargar el dataset usando la ruta completa y especificar la codificación y el separador
data = pd.read_csv('C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\data\\test_data3.0.csv', 
                   encoding='ISO-8859-1', sep=';')

# Limpiar nombres de columnas
data.columns = data.columns.str.strip().str.replace('ï»¿', '', regex=False)

# Separar las características (features) del objetivo (target)
X = data.drop('Orientacion_Vocacional', axis=1)  
y = data['Orientacion_Vocacional']  

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======== Random Forest ==========
random_forest = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15, min_samples_split=20)
random_forest.fit(X_train, y_train)

# Hacer predicciones y calcular precisión para Random Forest
y_pred_forest = random_forest.predict(X_test)
y_pred_proba_forest = random_forest.predict_proba(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f'Precisión del modelo de Random Forest: {accuracy_forest * 100:.2f}%')

# Guardar el modelo de Random Forest
joblib.dump(random_forest, 'C:\\Users\\gsucf\\Desktop\\DescubreTuTalento\\machine-learning\\modelos\\modelo_random_forest.pkl')

# Validación cruzada para Random Forest
scores_forest = cross_val_score(random_forest, X, y, cv=5)
print(f'Precisión media de validación cruzada con Random Forest: {scores_forest.mean() * 100:.2f}%')

# ======== Curvas ROC para cada clase ========
# Binarizar las etiquetas para la curva ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Generar curva ROC para cada clase
fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba_forest[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Each Class - Random Forest')
plt.legend(loc='best')
plt.show()

# ======== Mostrar la Matriz de Confusión ========
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Mostrar la matriz de confusión para Random Forest
plot_confusion_matrix(y_test, y_pred_forest, "Random Forest")

# ======== Curva de Error en Validación Cruzada ========
# Calcular error en cada fold de validación cruzada
error_rates = 1 - scores_forest
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), error_rates, marker='o', linestyle='--', color='b')
plt.xlabel('Fold de Validación Cruzada')
plt.ylabel('Error')
plt.title('Curva de Error en Validación Cruzada - Random Forest')
plt.show()

# ======== Calcular otras métricas ========
accuracy_forest = accuracy_score(y_test, y_pred_forest)
precision_forest = precision_score(y_test, y_pred_forest, average='weighted')
f1_forest = f1_score(y_test, y_pred_forest, average='weighted')

# Imprimir las métricas
print(f'Random Forest Model Accuracy: {accuracy_forest * 100:.2f}%')
print(f'Weighted Precision: {precision_forest * 100:.2f}%')
print(f'Weighted F1 Score: {f1_forest * 100:.2f}%')



# Generar datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Validación cruzada para obtener las métricas en entrenamiento y validación
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for train_index, val_index in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Entrenamiento
    rf_model.fit(X_train_fold, y_train_fold)
    
    # Predicciones
    y_train_pred = rf_model.predict(X_train_fold)
    y_train_prob = rf_model.predict_proba(X_train_fold)[:, 1]
    y_val_pred = rf_model.predict(X_val_fold)
    y_val_prob = rf_model.predict_proba(X_val_fold)[:, 1]
    
    # Calcular precisión y pérdida para entrenamiento y validación
    train_accuracies.append(accuracy_score(y_train_fold, y_train_pred))
    val_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    train_losses.append(log_loss(y_train_fold, y_train_prob))
    val_losses.append(log_loss(y_val_fold, y_val_prob))

# Graficar la precisión y la pérdida
plt.figure(figsize=(12, 6))

# Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Fold')
plt.ylabel('Precisión')
plt.title('Training and Validation Accuracy (Random Forest)')
plt.legend()

# Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Fold')
plt.ylabel('Log Loss')
plt.title('Training and Validation Loss (Random Forest)')
plt.legend()

plt.tight_layout()
plt.show()