SELECT COUNT(PassengerId) FROM train WHERE Pclass IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE Name IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE sex IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE age IS NULL; # it can be null
SELECT COUNT(PassengerId) FROM train WHERE SibSp IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE Parch IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE Ticket IS NULL;
SELECT COUNT(PassengerId) FROM train WHERE Fare IS NULL or Fare = 0; # can be 0
SELECT COUNT(PassengerId) FROM train WHERE Cabin IS NULL; # can be null
SELECT COUNT(PassengerId) FROM train WHERE Embarked IS NULL; # can be null

SELECT SUM(Survived), SibSp
FROM train
GROUP BY SibSp
ORDER BY SibSp ASC;

SELECT SUM(Survived), Parch
FROM train
GROUP BY Parch
ORDER BY Parch ASC;
# variable for none, 1 or 2 and >=3

SELECT SUM(Survived), Parch, SibSp
FROM train
GROUP BY SibSp, Parch
ORDER BY SUM(Survived) DESC;
# people with no parch and no sib has a tendency

SELECT SUM(Survived), COUNT(PassengerId), Cabin, (SUM(Survived)/COUNT(PassengerId)) as Porcentage
FROM train
GROUP BY Cabin
ORDER BY Porcentage DESC;
# Classes B,D and E have more 70% of survival
# Classes C, F, and G have between 50% and 60% of survival
# Classes A has 46
# null classes have 30 % of survival

SELECT SUM(Survived) as Survivals, COUNT(PassengerID) as ID, Sex, (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Sex
ORDER BY Percentage DESC;
# 74% of the woman survived

SELECT Pclass, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Pclass
ORDER BY Percentage DESC;
# People with higher class tends to survival

SELECT Age, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Age
ORDER BY Age DESC;

SELECT Embarked, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Embarked
ORDER BY Percentage;

#
select avg(Survived) as Porcentage, count(PassengerId) as Number,
    (case
    when (Age >= 60) then 'old'
    when (Age < 60 and Age >= 40) then 'adult'
    when (Age < 40 and Age >= 20) then 'young adult'
    when (Age >= 7 and Age < 20) then 'child'
    when (Age < 7) then 'baby' else 'young adult'
        end ) as New_Age
from train
group by New_Age
order by Porcentage desc;

select count(PassengerId) as Number, avg(Survived) as Porcentege,
       (case
        when (Fare > 0 and Fare <= 50) then '(0,50]'
        when (Fare > 50 and Fare <= 100) then '(50,100]'
        when (Fare > 100 and Fare <= 200) then '(100,200]'
        when (Fare > 200) then '(200,inf]' else '0'
           end) as Fare_
from train
group by Fare_
order by Porcentege desc;


select avg(Survived) as Porcentege, count(PassengerId),
       (case
           when Name like '%Mr%' then 'MR'
           when Name like '%Miss.%' then 'MISS'
           when Name like '%Ms.%' then 'MISS'
           when Name like '%Master.%' then 'MASTER'
           when Name like '%Mrs%' then 'MRS'
           when Name like '%Don.%' then 'DON'
           when Name like '%Rev.%' then 'REV'
           when Name like '%Dr.%' then 'DR'
           when Name like '%Mlle%' then 'LADIES'
           when Name like '%Countess.%' then 'LADIES'
           when Name like '%Mme.%' then 'LADIES'
           else 'OTHERS'
           end ) as Title_
from train
group by Title_
order by Porcentege;

select Name, Survived
from train
where Name like '%Rev%'