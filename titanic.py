import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.drop(['Cabin','Name','Ticket'], axis = 1, inplace = True)

train.loc[train.Age.isnull(), 'Age'] =  train.groupby("Pclass").Age.transform('median')

train.loc[train.Embarked.isnull(), 'Embarked'] =  mode(train.Embarked)

train['Sex'][train['Sex']== 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1

train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] =='C'] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                        train['Survived'], test_size = 0.2, random_state = 2)

logisticRegression = LogisticRegression(max_iter = 10000, solver='lbfgs')
logisticRegression.fit(X_train,y_train)

predictions = logisticRegression.predict(X_test)

accuracy = (91+50)/(91+50+9+29)

print(accuracy)
