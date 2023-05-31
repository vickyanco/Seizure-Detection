# KNN Algoritmo

import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, recall_score, precision_score

# importo dataset
df = pd.read_csv("C:/Users/USUARIO/Documents/Ingenieria biomedica/Identificacion de modelos y mineria de datos/dataset_final.csv")
# chequeo
#print(df)

# reemplazo por datos dicotómicos
df['CON_O_SIN_CONVULSION'] = df['CON_O_SIN_CONVULSION'].replace({'convulsion': 1, 'sin_convulsion': 0})
# chequeo
#print(df)

# marco las etiquetas
X = df.iloc[:, 1:138]
y = df.iloc[:, 0]
# chequeo
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
# chequeo
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

# establezco rangos
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# defino el modelo
knn = KNeighborsClassifier(n_neighbors=13, p=2, metric='euclidean')
knn.fit(X_train, y_train)

# predecir los resultados
y_pred = knn.predict(X_test)
#print(y_pred)

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
