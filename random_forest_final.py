# Random forest algoritmo

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, recall_score, precision_score

# importo dataset
df = pd.read_csv("C:/Users/USUARIO/Documents/Ingenieria biomedica/Identificacion de modelos y mineria de datos/dataset_final.csv")
# chequeo
#print(df)

# reemplazo por datos dicotómicos
df['CON_O_SIN_CONVULSION'] = df['CON_O_SIN_CONVULSION'].replace({'convulsion': 1, 'sin_convulsion': 0})
# chequeo
#print(df)
#print(df.head())

#df['is_train'] = np.random.uniform(0, 1, len(df)) <= .80
# chequeo
#print(df.head())

#train, test = df[df['is_train']==True], df[df['is_train']==False]
# chequeo
#print('Numero de observaciones en la training data', len(train))
#print('Numero de observaciones en la testing data', len(test))

#features = df.columns[1:139]
# chequeo
#print(features)

#y = pd.factorize(train['CON_O_SIN_CONVULSION'])[0]
# chequeo
#print(y)

# marco las etiquetas
X = df.iloc[:, 1:138]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#print(X_train.shape)
#print(y_train.shape)

# ver
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# defino el modelo
#randomfo = RandomForestClassifier(n_estimators=30, max_features="auto", random_state=44)
randomfo = RandomForestClassifier(n_estimators=30, random_state=44)
randomfo.fit(X_train, y_train)

max_features_range = np.arange(5,20,1)
n_estimators_range = np.arange(50,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

grid = GridSearchCV(estimator=randomfo, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

#print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))

grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results.head()
#grid_results.to_excel("C:/Users/USUARIO/Documents/Ingenieria biomedica/Identificacion de modelos y mineria de datos/tuning_randomfo.xlsx", index=False)

grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
grid_contour


# predecir los resultados
y_pred = randomfo.predict(X_test)
"""


# veo la prediccion
#print(randomfo.predict(test[features]))
#print(randomfo.predict_proba(test[features])[10:20])

#print(test['CON_O_SIN_CONVULSION'].head)
#print(pd.crosstab(test['CON_O_SIN_CONVULSION'], preds))

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