# Titanic

This is a repository that predicts, based on the information given, if a person would survive the Titanic disaster.  
For analyses of the data python and SQL were used.  
**Warning:** The readme is still in development as so the code.  
**Site:** [Titanic](https://www.kaggle.com/c/titanic)  
**Source Code 1:** [Linear Regression](https://github.com/Jumaruba/Titanic_kaggle/blob/master/models/titanic.py)  
**Source Code 2:** [Genetic Programming](https://github.com/Jumaruba/Titanic_kaggle/blob/master/models/genetic.py)
**Score Linear Regression:** 0.7790  
**Score Genetic Programming:** 0.76076
  
__Atention:__ At this repository, there're many different codes for different tries. Only one or a few codes are going to be mentioned at this README. Also, this respository is in constant development. Time to time new algorithms and methods are applied to achieve better results and then this README file is changed.   
__Atention2:__ Consults to the data files were also done using sql in the file `consults_train.sql`  

# Code for linear Regression
## Import libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
```

## Load the files using pandas

```python
train = pd.read_csv("data/train.csv")   # train table
test = pd.read_csv("data/test.csv")     # test table
test_result = pd.read_csv("data/gender_submission.csv") # test result table
```

## Analysing data: null values

First, let's see what labels are missing information:

```python 
for table in [train, test]:
    print(train.isnull().sum())
```

For each command the result is something like: 

```python
Train columns with null values: 
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
----------

Test columns with null values:
 PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
``` 

And in the end we have that the following columns can be null: 
- Age (test and train)
- Cabin (test and train)
- Embarked (train)
- Fare (test)

So, the first step is to handle null values.

### What to do 

There are many ways to handle missing data. One of these ways is to drop the data from the table, but for this case, this is not a good idea. Run the following commands:
```sql 
SELECT COUNT(PassengerId) FROM train WHERE Cabin IS NULL; # can be null
```
or
```python
train['Cabin' ].isnull().sum().sum() #where train is our dataframe
```

The result says that 687 passengers have null Cabin. Then, dropping 687 passengers out of the table isn't a good idea at all, only in case you wanna lose the biggest part of your train data.  
 __A good solution__ is to __replace__ the null values for others.

- For categorical data in common to replace for the __mode__ or a __value that symbolizes null values__.
- For numerical data is common to replace for the __median__ or __median__ value.
  
  _obs: replace numerical data for the mean might be not a good idea since outliers can influence negatively the result._

```python
for table in [train, test]:
        table["Embarked"] = table["Embarked"].fillna("N")
        table["Age"].fillna(table["Age"].mean(), inplace=True)
        table["Fare"] = table["Fare"].fillna(table["Fare"].mean())
```

There're still other values that might be null, but we're going to treat them on the next step.

## Change categorical data

Since we are going to use linear regression to the prediction, we need to substitute the categorical data for values.
Let's take a look at the _Embarked_ columns for example: 


```python 
print(train.groupby(['Embarked'])['Survived'].mean())
```
OUPUT: 
```python 

Embarked
C    0.553571
N    1.000000
Q    0.389610
S    0.336957
Name: Survived, dtype: float64
```

Ordering by desc order and substituting the categorical variables for numbers, we would have that: 
- N : 4 
- C : 3
- Q : 2 
- S : 1

Creating a new columns called `Embarked_n` we set it's values as followed: 

```python
# Embarked variables
for table in [train, test]: 
    embarked = table["Embarked"]
    table["Embarked_n"] = np.select([embarked == 'N', embarked == 'C', embarked == 'Q', embarked == 'S'],
                                    [4, 3, 2, 1], default=1)
```

Now, transforming ages into categoricals and doing the same process done before:
```python 
for table in [train, test]: 
    # Age variable
    age = table['Age']
    table['Age'] = np.select([age >= 60, np.logical_and(age < 60, age >= 40), np.logical_and(age < 40, age >= 20), np.logical_and(age >= 7, age < 20), age < 7], [1, 4, 2, 3, 5], default=2)

    # Fare variable
    fare = table['Fare']
    table['Fare_'] = np.select(
        [np.logical_and(fare > 100, fare <= 200), fare > 200, np.logical_and(fare > 50, fare <= 100),
            np.logical_and(fare > 0, fare <= 50), fare.isnull()], [5, 4, 3, 2, 1])

    # Sex variable
    table["is-woman"] = np.where(table["Sex"] == "female", True, False)

     # Variable for tickets
    ticket = table['Ticket']
    pattern_SC = "^(SC)"
    pattern_A = "^(A)"
    pattern_PC = "^(PC)"
    pattern_STON = "^(SOTON)|^(STON)"
    pattern_length_4 = "^[0-9]{3,4}$"
    pattern_length_5 = "^[0-9]{5}$"
    pattern_length_6 = "^[0-9]{6}$"
    pattern_length_7 = "^[0-9]{7,9}$"
    table['ticket_new'] = np.select(
        [ticket.str.contains(pattern_STON, regex=True), ticket.str.contains(pattern_SC, regex=True),
            ticket.str.contains(pattern_A, regex=True),
            ticket.str.contains(pattern_PC, regex=True), ticket.str.contains(pattern_length_4, regex=True),
            ticket.str.contains(pattern_length_5, regex=True),
            ticket.str.contains(pattern_length_6, regex=True), ticket.str.contains(pattern_length_7, regex=True)],
        [3, 6, 1, 8, 5, 7, 4, 2], default=4)

    # Cabin variables
    table["hasCabin"] = np.where(np.logical_or(table['Cabin'].isnull(), table['Cabin'].isna()), False, True)
```

## Delete the unused columns

```python 
table.drop(["Cabin", "Embarked", "Sex", "SibSp", "Ticket", 'Fare', 'Parch','Name'], axis=1, inplace=True)
```

Now, checking the result after all the transformations: 
```python 
for table in [train, test]: 
    print(table.head().to_string())
```

```python 
   PassengerId  Survived  Pclass  Age  Embarked_n  isAlone  Fare_  is-woman  ticket_new  hasCabin
0            1         0       3    2           1    False      2     False           1     False
1            2         1       1    2           3    False      3      True           8      True
2            3         1       3    2           1     True      2      True           3     False
3            4         1       1    2           1    False      3      True           4      True
4            5         0       3    2           1     True      2     False           4     False
   PassengerId  Pclass  Age  Embarked_n  isAlone  Fare_  is-woman  ticket_new  hasCabin
0          892       3    2           2     True      2     False           4     False
1          893       3    4           1    False      2      True           4     False
2          894       2    1           2     True      2     False           4     False
3          895       3    2           1     True      2     False           4     False
4          896       3    2           1    False      2      True           2     False
```
Now that we have cleaned the data, we can apply our model of linear regression: 

### Applying linear regression 

```python 
def model_linearRegression():
    train_x = train[train.columns[1:]]      #Getting all columns except for the passengerId
    train_y = train[train.columns[0:1]]     #Getting just the column of Survived
    test_x = test[test.columns[1:]]         
    model = LinearRegression().fit(train_x, train_y)    #Training the model 
    prediction_1 = pd.DataFrame(model.predict(test_x), columns=['Survived']) #Predict the result
    prediction_1 = prediction_1.round()     #Rounding the result, but it's still float
    frames = [prediction_1, test[test.columns[:1]]]
    result = frames[1].merge(frames[0], left_index=True, right_index=True)  #Adding the PassengerId to the talbe
    result = result.astype({'Survived': 'int'}) #Changing the survived table to integer
    result.to_csv("result_3.csv", index=False) #Transforming the result in cvs
```

# Code for genetic programming

Before readen this section, it's important you know some concepts about genetic programming. If you don't, it's recommended read the first three chapters of this book: [A Field Guide to
Genetic Programming](http://www0.cs.ucl.ac.uk/staff/ucacbbl/ftp/papers/poli08_fieldguide.pdf).  
Also, this code was inspired in this [notebook](https://www.kaggle.com/guesejustin/91-genetic-algorithms-explained-using-geap).  

## Libraries used 

- [Deap](https://deap.readthedocs.io/en/master/): library to build your genetic program. 
- Pandas 
- Numpy 

```python
import operator 
import math 
import random 
from deap import algorithms 
from deap import gp 
from deap import creator 
from deap import base 
from deap import tools 
import pandas as pd 
import numpy as np
```

