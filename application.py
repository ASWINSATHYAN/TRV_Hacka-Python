import flask
from flask import request, jsonify
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import json

app = flask.Flask(__name__)
df = pd.read_csv("train_data.csv")
df_X = df.iloc[:, 1:37].copy()  # Train Input
df_Y = df.iloc[:, 37:43].copy()  # Train Output
coursesArray = []
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(df_X, df_Y)
def findCourse(index):
    courses = {
        0: "G1",
        1: "G2",
        2: "P1",
        3: "P2",
        4: "D1",
        5: "D2"
    }
    coursesArray.append(courses.get(index, "NA"))

def getCourse():
    df = pd.read_csv("data.csv")
    result = multi_target_forest.predict(df)
    for x in result:
        for i, y in enumerate(x):
            if (y == 1):
                findCourse(i)
    return coursesArray
@app.route('/')
def hello_world():
    return 'Hey its Python Flask application!'

@app.route('/predictCourse', methods=['POST'])
def predictCourse():
    coursesArray.clear();
    applicationData =  json.loads(request.data)
    for data in applicationData.keys():
        if isinstance(applicationData[data], bool):
            if(applicationData[data]):
                applicationData[data] = 1
            else:
                applicationData[data] = 0
    with open("data.csv", 'r') as resultFile:
        lines = list(resultFile)

    if (len(lines)) > 0:
        values = lines[2].split(",")
        for index, data in enumerate(list(applicationData.values())):
            values[index] = data
        applicationData = values

    with open("data.csv", 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        if isinstance(applicationData, dict):
            wr.writerow(applicationData.keys())
            wr.writerow(applicationData.values())
        else:
            wr.writerow(json.loads(request.data).keys())
            wr.writerow(applicationData)
    predictedCourses = getCourse()
    response = {"Courses": predictedCourses}
    return jsonify(**response);

@app.route('/sample', methods=['POST'])
def api_response():
    return "Hi"

if __name__ == '__main__':
    app.debug = True
    app.run()
