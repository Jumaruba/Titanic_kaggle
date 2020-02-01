import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_result = pd.read_csv("data/gender_submission.csv")


def data_study():
    print(train
          ['Cabin'].isnull().sum())
    print(train['Age'].mean())
    corr = train.corr()
    print(corr.to_string())
    train_grp = train.groupby('Fare')['Survived'].mean()
    train_grp.plot()
    plt.show()
    plt.savefig("graphics/fare_survived.png")
    # sns.heatmap(corr, cmap=sns.diverging_palette(20, 220, n=200))
    # plt.savefig("graphics/heat_map.png")


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
        print(table.head().to_string())

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
    result.to_csv("result_2.csv", index=False)
    print(model.score(train_x, train_y))


data_study()
data_clean()
model()
