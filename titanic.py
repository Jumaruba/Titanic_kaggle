import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_result = pd.read_csv("data/gender_submission.csv")



# conclusion: NA values: Age, Cabin, Embarked, Fare
# setting the age
def data_clean():
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
        table['FamilySize'] = parch + sib + 1

        # Variables for alone
        table['isAlone'] = np.where(sib + parch == 0, True, False)

        # Variable age
        age = table['Age']
        table['Age'] = np.select([age >= 60, np.logical_and(age < 60, age >= 40), np.logical_and(age < 40, age >= 20),
                                  np.logical_and(age >= 7, age < 20), age < 7], [1, 4, 2, 3, 5], default=2)

        # Variable title

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
        table['name-length'] = table['Name'].apply(lambda x: len(x))

        # Cabin variables
        table["hasCabin"] = np.where(np.logical_or(table['Cabin'].isnull(), table['Cabin'].isna()), False, True)

        table["Fare"] = table["Fare"].fillna(table["Fare"].mean())
        table.drop(["Cabin", "Embarked", "Sex", "SibSp", "Ticket", 'Fare'], axis=1, inplace=True)
        # drop for now
        table.drop(["Name"], axis=1, inplace=True)



def model():

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
    print(model.score(train_x, train_y))


data_clean()
model()
