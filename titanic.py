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






# -----submission------

test.drop(['Cabin','Name','Ticket'], axis = 1, inplace = True)

test.loc[test.Age.isnull(), 'Age'] =  test.groupby("Pclass").Age.transform('median')
test.loc[test.Fare.isnull(), 'Fare'] =  mode(test.Fare)


test['Sex'][test['Sex']== 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1

test['Embarked'][test['Embarked'] == 'S'] = 0
test['Embarked'][test['Embarked'] =='C'] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

logisticRegression = LogisticRegression(max_iter = 10000, solver='lbfgs')
logisticRegression.fit(X = train.drop('Survived', axis =1),y = train['Survived'])
test['Survived'] = logisticRegression.predict(test)
test[['PassengerId', 'Survived']].to_csv('kaggle_submission1.csv', index = False)

# ------------------------------

# using feature engineering and cross validation and hyper parameter tuning
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
import re
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test['Survived'] = np.nan
full_data = pd.concat([train,test])

# print(full_data.head())
# Let's calculate percentages of missing values!
# print(full_data.isnull().mean().sort_values(ascending = False))

full_data["Embarked"] = full_data["Embarked"].fillna(mode(full_data["Embarked"]))

full_data["Sex"][full_data["Sex"] == "male"] = 0
full_data["Sex"][full_data["Sex"] == "female"] = 1


full_data["Embarked"][full_data["Embarked"] == "S"] = 0
full_data["Embarked"][full_data["Embarked"] == "C"] = 1
full_data["Embarked"][full_data["Embarked"] == "Q"] = 2

# sns.heatmap(full_data.corr(), annot = True)
# plt.show()

# print( full_data.groupby("Pclass")['Age'].sum())

full_data['Age'] = full_data.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

full_data['Fare'] = full_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

# full_data['Cabin'] = full_data['Cabin'].fillna('U')



# # Extract (first) letter!
# full_data['Cabin'] = full_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

# #converting categorial values into numeric
# cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
# full_data['Cabin'] = full_data['Cabin'].map(cabin_category)
# #print(list(full_data.Cabin.unique()))

# # Extract the salutation!
# full_data['Title'] = full_data.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

# Engineer 'familySize' feature
full_data['familySize'] = full_data['SibSp'] + full_data['Parch'] + 1

full_data.drop(['Cabin','Name','SibSp','Parch','Ticket'],axis=1, inplace=True)

# Recover test and train dataset
test = full_data.loc[full_data['Survived'].isna(),:]
test.drop(['Survived'],axis=1,inplace=True)

train = full_data.loc[full_data['Survived'].notna(),:]

# Cast 'Survived' back to integer
train['Survived'].astype(np.int8)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis = 1), train['Survived'], test_size = 0.2, 
                                                    random_state = 2)


# logisticRegression = LogisticRegression(max_iter = 10000)

# logisticRegression.fit(X_train,y_train)
# predictions = logisticRegression.predict(X_test)

# print(predictions.astype(np.int8))

# print(confusion_matrix(y_test,predictions))

# accuracy = (89+50)/(89+50+29+11)

# print(round(accuracy,2))

# **using crossvalidation**
# kf= KFold(n_splits=5, random_state=2)
# print(cross_val_score(logisticRegression, X_test, y_test, cv = kf).mean())


# ** USING RANDOM FOREST
#Initialize randomForest
randomForest = RandomForestClassifier(random_state = 2)
param_grid = {
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}


# Grid search
randomForest_CV = GridSearchCV(estimator = randomForest, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)

# Print best hyperparameters
# print(randomForest_CV.best_params_)
randomForestFinalModel = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 300)
randomForestFinalModel.fit(X_train, y_train)

# Predict!
predictions = randomForestFinalModel.predict(X_test)
from sklearn.metrics import accuracy_score

# Calculate the accuracy for our powerful random forest!
# print("accuracy is: ", round(accuracy_score(y_test, predictions), 2))

# Predict!
test['Survived'] = randomForestFinalModel.predict(test.drop(['PassengerId'], axis = 1))

# Write test predictions for final submission
test[['PassengerId', 'Survived']].to_csv('kagglesubmission3.csv', index = False)


****---Using different Modles and comparing the results---**
logisticRegression = LogisticRegression(max_iter = 10000, random_state=2)
logisticRegression.fit(X_train,y_train)
predictions = logisticRegression.predict(X_test)

print(accuracy_score(y_test,predictions))
# print(classification_report(y_test,predictions))

kneighborsclassifier = KNeighborsClassifier(n_neighbors=6)
kneighborsclassifier.fit(X_train,y_train)
predictions = kneighborsclassifier.predict(X_test)
print(accuracy_score(y_test,predictions))
# print(classification_report(y_test,predictions))

svm = SVC(kernel='linear',random_state=2)
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
print(accuracy_score(y_test,predictions))

svm = SVC(kernel='rbf',random_state=2)
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
print(accuracy_score(y_test,predictions))

svm = SVC(random_state=2)
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
print(accuracy_score(y_test,predictions))

randomForest = RandomForestClassifier(random_state = 2)
randomForest.fit(X_train,y_train)
predictions = randomForest.predict(X_test)
print(accuracy_score(y_test,predictions))

***-- got high accuracy in random forest --**

