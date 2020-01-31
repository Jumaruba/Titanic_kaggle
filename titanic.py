import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_result = pd.read_csv("data/gender_submission.csv")


def data_study():
    corr = train.corr()
    heat = sns.heatmap(corr, cmap=sns.diverging_palette(20, 220, n=200))
    plt.savefig("graphics/heat_map.png")


# conclusion: NA values: Age, Cabin, Embarked, Fare
# setting the age
def data_clean():
    for table in [train, test]:
        table["Embarked"] = table["Embarked"].fillna("C")
        table["Age"].fillna(table["Age"].mean(), inplace=True)
        table['Cabin'] = table["Cabin"].fillna("N")

        # Embarked variables
        table["EmbarkedC"] = np.where(table["Embarked"] == 'C', True, False)
        table["EmbarkedQ"] = np.where(table["Embarked"] == 'Q', True, False)
        table["EmbarkedS"] = np.where(table["Embarked"] == 'S', True, False)

        # Variables for classes
        table["Class1"] = np.where(table["Pclass"] == 1, True, False)
        table["Class2"] = np.where(table["Pclass"] == 2, True, False)
        table["Class3"] = np.where(table["Pclass"] == 3, True, False)

        # Variables for siblings
        table["Sib>1"] = np.where(table["SibSp"] > 1, True, False)  # variable more than one sibling
        table["noSib"] = np.where(table["SibSp"], False, True)

        # Treat Sex variable
        table["is-woman"] = np.where(table["Sex"] == "female", True, False)

        # Variables for parch
        table["noParch"] = np.where(table["Parch"] == 0, True, False)
        table["Parch1or2"] = np.select([table["Parch"] == 1, table["Parch"] == 2], [True, True], default=False)
        table["Parch>2"] = np.where(table["Parch"] > 2, True, False)

        # # Relation no parch and no sib
        # table["noSibParch"] = np.where(np.logical_and(table["noParch"], table["noSib"]), True, False)

        # Cabin variables
        table["CabinBDE"] = np.select(
            [table["Cabin"].str.contains('B'), table["Cabin"].str.contains("D"), table["Cabin"].str.contains("E")],
            [True, True, True], default=False)
        table["CabinCFG"] = np.select(
            [table["Cabin"].str.contains("C"), table["Cabin"].str.contains("F"), table["Cabin"].str.contains("G")],
            [True, True, True], default=False)
        table["CabinA"] = np.where(table["Cabin"].str.contains("A"), True, False)
        table["CabinTN"] = np.where(np.logical_or(table["Cabin"].str.contains("T"), table["Cabin"].str.contains("N")),
                                    True, False)

        table["Fare"] = table["Fare"].fillna(table["Fare"].mean())
        table.drop(["Cabin", "Embarked", "Parch", "Sex", "Pclass", "SibSp"], axis=1, inplace=True)
        # drop for now
        table.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)


def model():
    test_result.drop(["PassengerId"], axis=1, inplace=True)
    train_x = train[train.columns[1:]]
    train_y = train[train.columns[:1]]
    test_x = test[test.columns[:]]
    test_y = test_result[test_result.columns[:]]
    model = LinearRegression().fit(train_x, train_y)
    prediction_1 = model.predict(test_x)
    print(model.score(test_x, test_y))


data_clean()
data_study()
# model()
