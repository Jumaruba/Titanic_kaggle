import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
test_result = pd.read_csv("./data/gender_submission.csv")


# conclusion: NA values: Age, Cabin, Embarked, Fare
# setting the age
def data_clean():
    print(train.head().to_string())
    for table in [train, test]:
        table["Embarked"] = table["Embarked"].fillna("N")
        table["Age"].fillna(table["Age"].mean(), inplace=True)

        # Embarked variables
        embarked = table["Embarked"]
        table["Embarked_n"] = np.select([embarked == 'N', embarked == 'C', embarked == 'Q', embarked == 'S'],
                                        [4, 3, 2, 1], default=1)

        # Variables for family
        parch = table['Parch']
        sib = table['SibSp']
        #table['FamilySize'] = parch + sib + 1

        # Variables for alone
        table['isAlone'] = np.where(sib + parch == 0, True, False)

        # Variable age
        age = table['Age']
        table['Age'] = np.select([age >= 60, np.logical_and(age < 60, age >= 40), np.logical_and(age < 40, age >= 20),
                                  np.logical_and(age >= 7, age < 20), age < 7], [1, 4, 2, 3, 5], default=2)

        # Variable for fare
        fare = table['Fare']
        table['Fare_'] = np.select(
            [np.logical_and(fare > 100, fare <= 200), fare > 200, np.logical_and(fare > 50, fare <= 100),
             np.logical_and(fare > 0, fare <= 50), fare.isnull()], [5, 4, 3, 2, 1])
        # Treat Sex variable
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

        # Name length
        # table['name-length'] = table['Name'].apply(lambda x: len(x))

        # Cabin variables
        table["hasCabin"] = np.where(np.logical_or(table['Cabin'].isnull(), table['Cabin'].isna()), False, True)

        table["Fare"] = table["Fare"].fillna(table["Fare"].mean())
        table.drop(["Cabin", "Embarked", "Sex", "SibSp", "Ticket", 'Fare', 'Parch'], axis=1, inplace=True)
        # drop for now
        table.drop(["Name"], axis=1, inplace=True)
        print(table.head().to_string())


def model_linearRegression():
    # score 0.7770
    # Output the model
    train_x = train[train.columns[2:]]
    train_y = train[train.columns[1:2]]
    test_x = test[test.columns[1:]]
    test_y = test_result[test_result.columns[1:]]
    model = LinearRegression().fit(train_x, train_y)
    prediction_1 = pd.DataFrame(model.predict(test_x), columns=['Survived'])
    prediction_1 = prediction_1.round()
    frames = [prediction_1, test[test.columns[:1]]]
    result = frames[1].merge(frames[0], left_index=True, right_index=True)
    result = result.astype({'Survived': 'int'})
    result.to_csv("result_3.csv", index=False)

    # To print how good is how good is out model
    print("--- LINEAR REGRESSION ---")
    x_train, x_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2)
    model = LinearRegression().fit(x_train[x_train.columns[2:]], y_train)
    x_test = x_test[x_test.columns[2:]]
    print(model.score(x_test, y_test))


def model_knn():
    # score 0.65550
    train_x = train[train.columns[2:]]
    train_y = train[train.columns[1:2]]
    test_x = test[test.columns[1:]]
    test_y = test_result[test_result.columns[1:]]
    knn = KNeighborsClassifier().fit(train_x, train_y)
    knn_prediction = pd.DataFrame(knn.predict(test_x), columns=['Survived'])
    knn_prediction = knn_prediction.round()
    frames = [knn_prediction, test[test.columns[:1]]]
    result = frames[1].merge(frames[0], left_index=True, right_index=True)
    result.to_csv("result_knn.csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2)

    print("--- KNN ---")
    x_train, x_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2)
    model = KNeighborsClassifier().fit(x_train[x_train.columns[2:]], y_train)
    x_test = x_test[x_test.columns[2:]]
    print(model.score(x_test, y_test))


def randomForest():
    train_x = train[train.columns[2:]]
    train_y = train[train.columns[1:2]]
    test_x = test[test.columns[1:]]
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1).fit(train_x, train_y)
    prediction = model.predict(test_x)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})
    output.to_csv('RandomForest_result1.csv', index=False)


data_clean()
model_linearRegression()
model_knn()
randomForest()
