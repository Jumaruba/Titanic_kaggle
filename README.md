# Titanic 

This is a repository that predicts, based on the information given, if a person would survive the titanic disaster.  
For analyses of the data python and SQL were used. 
## Analysing data: null values

First let's see what labels are missing information: 


```sql=
select COUNT(PassengerId) from test where Pclass is null;
select COUNT(PassengerId) from test where Name is null;
select COUNT(PassengerId) from test where sex is null;
select COUNT(PassengerId) from test where age is null; # it can be null
select COUNT(PassengerId) from test where SibSp is null;
select COUNT(PassengerId) from test where Parch is null;
select COUNT(PassengerId) from test where Ticket is null;
select COUNT(PassengerId) from test where Fare is null or Fare = 0; # can be 0
select COUNT(PassengerId) from test where Cabin is null; # can be null
select COUNT(PassengerId) from test where Embarked is null; # can be null
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

