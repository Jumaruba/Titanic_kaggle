# Titanic 

This is a repository that predicts, based on the information given, if a person would survive the titanic disaster.  
For analyses of the data python and SQL were used. 
## Analysing data: null values

First let's see what labels are missing information: 


```sql
SELECT COUNT(PassengerId) FROM test WHERE Pclass IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE Name IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE sex IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE age IS NULL; # it can be null
SELECT COUNT(PassengerId) FROM test WHERE SibSp IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE Parch IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE Ticket IS NULL;
SELECT COUNT(PassengerId) FROM test WHERE Fare IS NULL or Fare = 0; # can be 0
SELECT COUNT(PassengerId) FROM test WHERE Cabin IS NULL; # can be null
SELECT COUNT(PassengerId) FROM test WHERE Embarked IS NULL; # can be null
```

For each command the result is none or something like: 

![](https://i.imgur.com/xm9KWaV.png)

The same analyses must be done for the test table.
And in the end we have that the following columns can be null: 
- Age (test and train)
- Cabin (test and train)
- Embarked (train)
- Fare (test)

So, the first step is to handle this null values. 

