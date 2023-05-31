import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, recall_score, precision_score

# importo dataset
df = pd.read_csv("C:/Users/USUARIO/Documents/Ingenieria biomedica/Identificacion de modelos y mineria de datos/dataset_final.csv")

# reemplazo por datos dicotómicos
df['CON_O_SIN_CONVULSION'] = df['CON_O_SIN_CONVULSION'].replace({'convulsion': 1, 'sin_convulsion': 0})

# marco las etiquetas
X = df.iloc[:, 1:138]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# estandarizo mis datos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# inicializo ANN
ann = tf.keras.models.Sequential()

# Agrego First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# Agrego Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

# units:- number of neurons that will be present in the respective layer
# activation:- specify which activation function to be used. rectified linear unit

# Agrego Output Layer
# In a binary classification problem (only two classes) as output (1 and 0)
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

# Compiling ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# Fitting ANN
ann.fit(X_train,y_train,batch_size=32,epochs = 100)

# predecir los resultados
y_pred = ann.predict(X_test)
"""
# evaluo el modelo

# matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print('matriz de confusión: ')
print(cm)

# f1
print('F1 score: ', f1_score(y_test, y_pred))

# accuracy
print('Accuracy: ', accuracy_score(y_test, y_pred))

# ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# SENSIBILITY o RECALL
sensitivity = recall_score(y_test, y_pred)
print("Sensitivity/Recall:", sensitivity)

# PRECISION
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# SPECIFITY
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)
"""