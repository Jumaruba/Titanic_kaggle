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
ORDER BY SUM(Survived) DESC
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
ORDER BY Percentage DESC
# 74% of the woman survived

SELECT Pclass, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Pclass
ORDER BY Percentage DESC
# People with higher class tends to survival

SELECT Age, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Age
ORDER BY Age DESC;

SELECT Embarked, SUM(Survived), COUNT(PassengerId), (SUM(Survived)/COUNT(PassengerId)) as Percentage
FROM train
GROUP BY Embarked
ORDER BY Percentage



