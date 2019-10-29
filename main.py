import numpy as np
import pandas as pd
import os
from string import Template
from flask import Flask
from itertools import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, static_folder="assets")

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

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


def generateRows(result, names):
    returner = []
    for row, name in zip(result.itertuples(index=True), names):
        string = "<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>".format(row[0][0], name, row[1])
        returner.append(string)

    return "".join(row for row in returner)

df['Age'] = df[['Age', 'Pclass']].apply(mean_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(mean_age, axis=1)

age_range = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in df['Age']:
    age_range[int(i//10)] = age_range[int(i//10)] + 1

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

passenger_ids = test['PassengerId']
names = test['Name']

test.drop(['Sex', 'Embarked', 'Ticket', 'Name', 'PassengerId', 'Parch', 'Pclass'], axis=1, inplace=True)

X = df.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Parch',
         'PassengerId', 'Pclass', 'Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=101)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

predictions = knn.predict(test)

result = pd.DataFrame(
    data=predictions, index=[passenger_ids, names], columns=['Survived'])

relationships = df.SibSp.value_counts().to_dict()
relation_labels = list(relationships.keys())
relation_vals = list(relationships.values())


params = {
    'maleCount': len(df[(df.Sex == "male") & (df.Survived == 1)]),
    'femaleCount': len(df[(df.Sex == "female") & (df.Survived == 1)]),
    'firstClass': len(df[(df.Pclass == 1) & (df.Survived == 1)]),
    'secondClass': len(df[(df.Pclass == 2) & (df.Survived == 1)]),
    'thirdClass': len(df[(df.Pclass == 3) & (df.Survived == 1)]),
    'relationshipLabels': str(relation_labels),
    'relationshipValues': str(relation_vals),
    'ageRange': str(age_range),
    'rowsWithData': generateRows(result, names)
}

template = Template('''
        <!DOCTYPE HTML>
<html>
    <head>
        <title>Titanic</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.3/Chart.min.js"></script>
        <link rel="stylesheet" href="assets/css/main.css" />
        <noscript><link rel="stylesheet" href="assets/css/noscript.css" /></noscript>
    </head>
    <body class="is-preload">

        <!-- Wrapper -->
            <div id="wrapper">

                <!-- Header -->
                    <header id="header" class="alt">
                        <h1>Titanic</h1>
                        <p>Passenger Survival Predictor using KNN Algorithm</p>
                    </header>

                <!-- Nav -->
                    <nav id="nav">
                        <ul>
                            <li><a href="#gender">Gender</a></li>
                            <li><a href="#relationship">Relationship</a></li>
                            <li><a href="#pclass">Class</a></li>
                            <li><a href="#age">Age</a></li>
                            <li><a href="#results">Result</a></li>
                        </ul>
                    </nav>

                <!-- Main -->
                    <div id="main">
        
                        <section id="gender" class="main special">
                            <header class="major">
                                <h2>Gender Distribution</h2>
                                <p>Most of the survivors were females</p>
                            </header>

                            <canvas id="genderCanvas"></canvas>
                        </section>

                        <section id="relationship" class="main special">
                            <header class="major">
                                <h2>Relationship Distribution</h2>
                                <p>Survival rate is inversely proportional to number of relatives on board</p>
                            </header>

                            <canvas id="relationCanvas"></canvas>
                        </section>

                        <section id="pclass" class="main special">
                            <header class="major">
                                <h2>Passenger Class Distribution</h2>
                                <p>Highest number of First class passengers survived</p>
                            </header>

                            <canvas id="pclassCanvas"></canvas>
                        </section>

                        <section id="age" class="main special">
                            <header class="major">
                                <h2>Age Distribution</h2>
                                <p>Most survivors were in their 20-30s</p>
                            </header>

                            <canvas id="ageCanvas"></canvas>
                        </section>

                        <section id="results" class="main special">
                            <header class="major">
                                <h2>Results</h2>
                            </header>

                            <div class="table-wrapper">
                                <table>
                                    <tr>
                                        <th>Passenger Id</th>
                                        <th>Name</th>
                                        <th>Surivived?</th>
                                    </tr>
                                    $rowsWithData
                                </table>
                            </div>
                        </section>

                    </div>

                    <footer id="footer">
                        <section>
                            <h2>Made by</h2>
                            <p>By <b>Tanishq Jain</b> (16101A0015), <b>Sanket Udapi</b> (16101A0014) and <b>Amey Nikam</b> (16101A0017)</p>
                        </section>
                        <section>
                            <h2>Dataset</h2>
                            <p><a href="https://www.kaggle.com/c/titanic/data">Available here</a></p>
                        </section>
                    </footer>

            </div>

        <!-- Scripts -->
            <script src="assets/js/jquery.min.js"></script>
            <script src="assets/js/jquery.scrollex.min.js"></script>
            <script src="assets/js/jquery.scrolly.min.js"></script>
            <script src="assets/js/browser.min.js"></script>
            <script src="assets/js/breakpoints.min.js"></script>
            <script src="assets/js/util.js"></script>
            <script src="assets/js/main.js"></script>
            <script>
                const gender = document.getElementById('genderCanvas').getContext('2d')

                new Chart(gender, {
                    type: 'pie',
                    data: {
                        labels: ['Males', 'Females'],
                        datasets: [{
                            label: 'Survivors',
                            data: [$maleCount, $femaleCount],
                            backgroundColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                            ],
                            borderWidth: 1
                        }]
                    }
                })

                const pclass = document.getElementById('pclassCanvas').getContext('2d')

                new Chart(pclass, {
                    type: 'doughnut',
                    data: {
                        labels: ['First', 'Second', 'Third'],
                        datasets: [{
                            label: 'Survivors',
                            data: [$firstClass, $secondClass, $thirdClass],
                            backgroundColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                            ],
                            borderWidth: 1
                        }]
                    }
                })

                const relation = document.getElementById('relationCanvas').getContext('2d')

                new Chart(relation, {
                    type: 'line',
                    data: {
                        labels: $relationshipLabels,
                        datasets: [{
                            label: "Survivors",
                            data: $relationshipValues,
                            backgroundColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)'
                            ],
                            borderWidth: 1
                        }]
                    }
                })

                const age = document.getElementById('ageCanvas').getContext('2d')

                new Chart(age, {
                    type: 'bar',
                    data: {
                        labels: ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
                        datasets: [{
                            label: "Survivors",
                            data: $ageRange,
                            backgroundColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)'
                            ],
                            borderWidth: 1
                        }]
                    }
                })


            </script>
    </body>
</html>
    ''').safe_substitute(params)


@app.route("/")
def index():
    return template

@app.route("/favicon.ico")
def favicon():
    return "False"

if __name__ == '__main__':
    app.run()