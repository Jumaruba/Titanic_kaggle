import pandas as pd 
import matplotlib.pyplot as plt 

train = pd.read_csv("./data/train.csv")

fig = plt.figure(figsize = (9,6))
print(train.count())
print(train.Cabin.value_counts())
print(train.Embarked.value_counts())

print(train.head().to_string())

#survived woman of each class 
plt.subplot2grid((3,4),(0,0))
for i in [1,2,3]: 
    train.Pclass[(train.Survived == 1) & (train.Sex == 'female')].value_counts(normalize  = True).plot(kind = "bar", alpha = 0.2)
plt.title("Women survived in each class")

#How many woman died in each class 
plt.subplot2grid((3,4), (0,1))
for i in [1,2,3]: 
    train.Pclass[(train.Survived == 0) & (train.Sex == 'female')].value_counts(normalize = True).plot(kind = "bar", alpha = 0.2)
plt.title("Women died in each class")

#How many poor woman died 
plt.subplot2grid((3,4),(0,2))
train.Survived[(train.Pclass == 3) & (train.Sex == 'female')].value_counts(normalize = True).plot(kind="bar", alpha = 0.2)
plt.title("How many poor women died")

#How many rich man survived
plt.subplot2grid((3,4), (0,3))
train.Survived[(train.Pclass == 1) & (train.Sex == 'male')].value_counts(normalize = True).plot(kind="bar", alpha = 0.2)
plt.title("How many rich men died")

#How many rich woman survived 
plt.subplot2grid((3,4), (1,0)) 
train.Survived[(train.Pclass == 1) & (train.Sex == 'female')].value_counts(normalize = True).plot(kind = "bar", alpha = 0.2 )
plt.title("How many rich women died")


    

plt.show()