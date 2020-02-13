import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
import pandas as pd 
def clean(train): 
    #cleaning the sex 
    train.loc[train.Sex == 'female', "Sex"] = 1 
    train.loc[train.Sex == 'male', "Sex"] = 0 

    train.Age = train.Age.fillna(train.Age.dropna().mean()) 
    train.Fare = train.Fare.fillna(train.Fare.dropna().mean())

    train.Embarked =  train.Embarked.fillna('S')
    train.loc[train.Embarked == 'S', "Embarked"] = 1 
    train.loc[train.Embarked == 'C', "Embarked"] = 2 
    train.loc[train.Embarked == 'Q', "Embarked"] = 3 
    
    

    train['Married'] = 0 
    train['Married'].loc[train.Name == 2] = 1 

    return train 
