from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier

Datos = load_iris()
x = Datos.data
y = Datos.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=0)
standar_scaler = StandardScaler()
x_train = standar_scaler.fit_transform(x_train)
x_test = standar_scaler.fit_transform(x_test)
#####################################
### Naive Byaes
NaiveB = GaussianNB()
NaiveB.fit(x_train, y_train)
NB_Pred = NaiveB.predict(x_test)

# Metricas
F1_Score_NB = f1_score(y_test, NB_Pred, average= 'micro')
Accuracy_NB = accuracy_score(y_test, NB_Pred)
MCC_NB = matthews_corrcoef(y_test, NB_Pred)

print ('F1 Naive Bayes:', F1_Score_NB.round(3))
print ('Accuracy Naive Bayes:', Accuracy_NB.round(3))
print ('MCC Naive Bayes:', MCC_NB.round(3))

########################################
### DecisionTree
Deci_Tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
Deci_Tree.fit(x_train, y_train)
DT_Pred = Deci_Tree.predict(x_test)

#Metricas
F1_Score_DT = f1_score(y_test, DT_Pred, average= 'micro')
Accuracy_DT = accuracy_score(y_test, DT_Pred)
MCC_DT = matthews_corrcoef(y_test, DT_Pred)

print (28*'-')
print ('F1 Decision Tree:', F1_Score_DT.round(3))
print ('Accuracy Decision Tree:', Accuracy_DT.round(3))
print ('MCC Decision Tree:', MCC_DT.round(3))

# gini y minimo de muestras en hoja 
Deci_Tree = DecisionTreeClassifier(criterion='gini', random_state=0, min_samples_leaf= 3)
Deci_Tree.fit(x_train, y_train)
DT_Pred = Deci_Tree.predict(x_test)

#Metricas
F1_Score_DT = f1_score(y_test, DT_Pred, average= 'micro')
Accuracy_DT = accuracy_score(y_test, DT_Pred)
MCC_DT = matthews_corrcoef(y_test, DT_Pred)

print (28*'-')

print ('F1 Decision Tree:', F1_Score_DT.round(3))
print ('Accuracy Decision Tree:', Accuracy_DT.round(3))
print ('MCC Decision Tree:', MCC_DT.round(3))