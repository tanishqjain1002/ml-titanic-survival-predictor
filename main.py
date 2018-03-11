import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

'''Empty data visualisation'''
'''A lot of data is missing from the Cabin column and an okish chunk of data missing from the Age column. Better to drop the Cabin column'''
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
df.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

'''Male Female distribution'''
'''Most of the survivors were females'''
sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()

'''Passenger Ticketwise distribution'''
'''Survival rate is in directly proportional to expensive tickets'''
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.show()

'''Age distribution'''
'''Maximum survivors were 20 to 30 yrs old also a good load of kids'''
sns.distplot(df['Age'].dropna(), kde=False, bins=50)
plt.show()

'''Relationship distribution'''
'''No. of relationships is inversely proportional to the Survival rate. Singles survive the most'''
sns.countplot(x='SibSp', data=df)
plt.show()

'''Fare distribution'''
'''Ticket prices were inversely proportional to Survival Rate'''
df['Fare'].hist(bins=100, figsize=(10, 4))
plt.show()


'''Filling missing ages with mean age of that Passenger class'''
sns.boxplot(x='Pclass', y='Age', data=df)
plt.show()


def mean_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


df['Age'] = df[['Age', 'Pclass']].apply(mean_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(mean_age, axis=1)


sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

'''Drop all remaining NaN'''
df.dropna(inplace=True)
test.dropna(inplace=True)

'''Remove Perfect predictors from dataframe'''
sex = pd.get_dummies(df['Sex'], drop_first=True)
embark = pd.get_dummies(df['Embarked'], drop_first=True)
sex_test = pd.get_dummies(test['Sex'], drop_first=True)
embark_test = pd.get_dummies(test['Embarked'], drop_first=True)

# making dummy because the column has only 3 values (1,2,3)
# TODO: maybe remove this dummy and not drop it 7 lines below if algorithm doesnt work
pclass = pd.get_dummies(df['Pclass'], drop_first=True)
pclass_test = pd.get_dummies(test['Pclass'], drop_first=True)

'''Modify Dataframe with training series'''
df = pd.concat([df, sex, embark, pclass], axis=1)
test = pd.concat([test, sex_test, embark_test, pclass_test], axis=1)

'''Remove Unnecessary columns'''
# TODO: dont drop Pclass if algorith fails
df.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Parch',
         'PassengerId', 'Pclass'], axis=1, inplace=True)

passenger_ids = test['PassengerId']

test.drop(['Sex', 'Embarked', 'Ticket', 'Name',
           'PassengerId', 'Parch', 'Pclass'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1, random_state=101)

lgr = LogisticRegression()

lgr.fit(X_train, y_train)

predictions = lgr.predict(test)

result = pd.DataFrame(
    data=predictions, index=passenger_ids, columns=['Survived'])

result.to_csv('results.csv', header=True)
