import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from numpy import std 
import seaborn as sns
from scipy import stats 



def data_study():
    train = pd.read_csv("data/train.csv")
    fare_survived = train[['Survived', 'Fare']]
    print(fare_survived.corr())
    #From this we can assume that the correlation between fare and survived is low 
    #Let's try to delete the outliers values... 
    print(fare_survived.describe()) #let's figure it out, outliers using the standard deviation 
    stand = std(np.array(fare_survived.Fare))
    print("Standard deviation:", stand)
    #More than 3 standard deviation is considered an outlier 
    
    train.Fare.where(np.abs(stats.zscore(train.Fare)) < 3, inplace = True) 
    fare_survived = train[['Survived', 'Fare']]
    print(fare_survived.corr()) #by removing the outliers, we have that the fare correlation has increased 
    # however, results are presenting to be worst 
    

data_study()
