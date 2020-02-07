import pandas as pd 
import numpy as np 
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier 

# let's import the data 

data_raw = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
test_raw = test.copy(deep = True)
# deep = True means that val has a different memory address of data_raw. It's a different object
train = data_raw.copy(deep = True)

frames = [train, test]

print(train.groupby(['Survived']))

# Let's findout the columns with null values
separator = '-'*10 + '\n'
print('Train columns with null values:\n', train.isnull().sum())
print(separator)
print('Test columns with null values:\n', test.isnull().sum())
print(separator)
print(data_raw.describe(include = 'all'))
print(separator)

### COMPLETING DATA
# The following columns have null values: age, cabin, embarked, fare 

# values = median(). Why not mean? Outliers might modify the results
# categorical = mode()
for dataset in frames: 
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True) 
    
# deleting columns from train 
drop = ['PassengerId','Cabin','Ticket']
train.drop(drop, axis = 1, inplace = True)

# Checking if everything is ok
print(train.isnull().sum())
print(separator)

### CREATE: Feature engineering

for dataset in frames: 
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1 
    dataset['isAlone'] = np.where(dataset['FamilySize'] == 1, True, False) 

    # Getting the title
    dataset['Title'] = dataset['Name'].str.split(",", expand = True)[1].str.split(".", expand=True)[0]

    # Continuos variables to bin 
    # qcut: equal number of elements in each division
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    # Continuos variables to bin 
    # cut: equal intervals
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


# Cleanup rare title names 
stat_min = 10  # minimun sample size
title_names = (train['Title'].value_counts() < stat_min)

train['Title'] = np.where(title_names[train['Title']], 'Misc', train['Title'] )
print(train['Title'].value_counts())
print(separator)

# Let's the statistics again 

print(train.describe())
print(train.info())
print(test.info())

### CONVERT FORMATS 

# Object variables: Name, Sex, Title, Embarked
# Categorical variables: FareBin, AgeBin 

le = preprocessing.LabelEncoder()
for dataset in frames: 
    dataset['Sex_Code'] = le.fit_transform(dataset['Sex'])
    dataset['Title_Code'] = le.fit_transform(dataset['Title'])
    dataset['FareBin_Code'] = le.fit_transform(dataset['FareBin'])
    dataset['AgeBin_Code'] = le.fit_transform(dataset['AgeBin'])
    dataset['Embarked'] = le.fit_transform(dataset['Embarked'])

print(train.head())
drop = ['Name','AgeBin','Sex','FareBin','Title','Fare','Age']
train.drop(drop, axis = 1, inplace = True)
test.drop(drop + ['Ticket','Cabin'], axis = 1, inplace = True)
print(test.head().to_string())
print(train.head().to_string())


### PREPARING DATA FOR PREDICTION
def model_linearRegression():
    # score 0.7770
    # Output the model
    train_x = train[train.columns[1:]]
    train_y = train[train.columns[0:1]]
    test_x = test[test.columns[1:]]
    model = LinearRegression().fit(train_x, train_y)
    prediction_1 = pd.DataFrame(model.predict(test_x), columns=['Survived'])
    prediction_1 = prediction_1.round()
    frames = [prediction_1, test[test.columns[:1]]]
    result = frames[1].merge(frames[0], left_index=True, right_index=True)
    result = result.astype({'Survived': 'int'})
    result.to_csv("result_3.csv", index=False)

def randomForest():
    train_x = train[train.columns[1:]]
    train_y = train[train.columns[0:1]]
    test_x = test[test.columns[1:]]
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(train_x, train_y)
    prediction = pd.DataFrame({'PassengerId': test_raw.PassengerId, 'Survived':clf.predict(test_x)})
    prediction.to_csv("result_random.csv", index=False)


#model_linearRegression()
randomForest()