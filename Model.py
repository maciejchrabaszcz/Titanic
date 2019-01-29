import numpy as np
import pandas as pd
import re
train_set = pd.read_csv('train.csv')
   
def change_titles(dataset):
    dataset['Title'] = dataset['Title'].replace(['Lady', 'theCountess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return dataset

def edit_data(train_set):
    title = []
    for name in train_set['Name']:
        name = name.split(',')[1]
        name = name.split('.')[0]
        name = re.sub(r"[ .,]", "", name)
        #name = name.lower()
        title.append(name)
        
    #train_set['Title'] = train_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    train_set['Title'] = title
    train_set['Title'].describe()
    train_set = train_set.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    #train_set = train_set.dropna(subset = ['Age', 'SibSp', 'Parch', 'Fare'])
    train_set.loc[train_set['Age'].isnull(), 'Age'] = train_set['Age'].mean()
    train_set.loc[train_set['Parch'].isnull(), 'Parch'] = train_set['Parch'].mean
    train_set.loc[train_set['Fare'].isnull(), 'Fare'] = train_set['Fare'].mean()
    train_set.loc[train_set['SibSp'].isnull(), 'SibSp'] = train_set['SibSp'].mean()
    train_set.loc[train_set['Age'] <= 16, 'Age'] = 0
    train_set.loc[(train_set['Age'] > 16) & (train_set['Age'] <= 32), 'Age'] = 1
    train_set.loc[(train_set['Age'] > 32) & (train_set['Age'] <= 48), 'Age'] = 2
    train_set.loc[(train_set['Age'] > 48) & (train_set['Age'] <= 64), 'Age'] = 3
    train_set.loc[train_set['Age'] >64, 'Age'] = 4
    pd.crosstab(train_set['Title'], train_set['Sex'])
    train_set = change_titles(train_set)
    train_set = pd.get_dummies(train_set, columns = ['Pclass', 'Sex', 'Embarked', 'Age', 'Title'], dummy_na = True, drop_first = True)
    return train_set

train_set = edit_data(train_set)

y = train_set.iloc[:, 1]
y = np.array(y)
X = train_set.iloc[:, 2:]
X = np.array(X)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#X = X_train
#y = y_train

model = Sequential()
model.add(Dense(units = 11, activation = 'relu', input_dim = 21))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X, y, batch_size = 10, epochs = 250)

#y_pred = model.predict(X_test)
#y_pred = (y_pred > 0.6)
#
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#
#accuracy = (cm[0,0] + cm [1,1])/cm.sum()
#print('Accuracy is: %.4f' % accuracy)

test = pd.read_csv('test.csv')
test = edit_data(test)
test_id = test.iloc[:,0]
test = test.iloc[:, 1:]

test = min_max_scaler.transform(test)
test_predict = model.predict(test)
test_predict = (test_predict > 0.6).astype(int)
test_predict = np.resize(test_predict, (418, 1))
#d = {'Survived' : test_predict}
test_predict = pd.DataFrame(test_predict, columns = ['Survived'])
predictions = pd.concat([test_id, test_predict], axis = 1)
predictions.to_csv('preditions.csv', index = False)