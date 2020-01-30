import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
set = [train, test]


def create_variables():
    for table in set:
        # Variables for classes
        table["Class1"] = np.where(table["Pclass"] == 1, True, False)
        table["Class2"] = np.where(table["Pclass"] == 2, True, False)
        table["Class3"] = np.where(table["Pclass"] == 3, True, False)
        table = table.drop(table["Pclass"])

        # Variables for siblings
        table["Sib>1"] = np.where(table["SibSp"] > 1, True, False)  # variable more than one sibling
        table["noSib"] = np.where(table["SibSp"], False, True)
        table = table.drop(columns=["SibSp"])  # drop siblings at the end

        # Treat Sex variable
        table["is-woman"] = np.where(table["Sex"] == "female", True, False)
        table = table.drop(columns="Sex")

        # Variables for parch
        table["noParch"] = np.where(table["Parch"] == 0, True, False)
        table["Parch1or2"] = np.select([table["Parch"] == 1, table["Parch"] == 2], [True, True], default=False)
        table["Parch>2"] = np.where(table["Parch"] > 2, True, False)
        table = table.drop(columns=["Parch"])

        # Relation no parch and no sib
        table["noSibParch"] = np.where(np.logical_and(table["noParch"], table["noSib"]), True, False)

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
        table = table.drop(columns="Cabin")



        print(table.head().to_string())


def fill_na():
    # conclusion: NA values: Age, Cabin, Embarked, Fare
    # setting the age
    for table in set:
        table["Age"].fillna(table["Age"].mean(), inplace=True)
        table['Cabin'] = table["Cabin"].fillna("N")

    # filling na values


fill_na()
create_variables()
