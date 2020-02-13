import pandas as pd 
import numpy as np 
import utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("./data/train.csv")
train = utils.clean(train) 
print(train.head().to_string())

target = train.Survived.values
features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Married']].values

classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)      #our prediction 
 
print(classifier_.score(features, target)) 

#now this is a curve 

poly = preprocessing.PolynomialFeatures(degree = 2)
poly_features = poly.fit_transform(features) 
classifier_ = classifier.fit(poly_features, target) 
print(classifier_.score(poly_features, target))

# Now applying to test 

test = pd.read_csv("./data/test.csv")
passengerId = test.PassengerId 
test = utils.clean(test) 
features_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Married']].values
poly_features = poly.fit_transform(features_test)
prediction = classifier_.predict(poly_features) 

result = pd.DataFrame({'PassengerId': passengerId, 'Survived': prediction})
result.to_csv("prediction.csv", index = False)

