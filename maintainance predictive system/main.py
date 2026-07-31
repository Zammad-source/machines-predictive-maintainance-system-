from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

app = Flask(__name__)

df = pd.read_csv('cleaned_dataset.csv')

X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
X.columns = X.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False)
y = df['Machine failure']

X['Power'] = X['Rotational speed rpm'] * X['Torque Nm']
X['Temp_diff'] = X['Process temperature K'] - X['Air temperature K']
X['Tool_Torque'] = X['Tool wear min'] * X['Torque Nm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

models = {
    'decision_tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'xgboost': XGBClassifier(eval_metric='logloss', random_state=42),
    'lightgbm': LGBMClassifier(class_weight='balanced', random_state=42)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    trained_models[name] = model
    joblib.dump(model, f"{name}_model.pkl")

print("\nAll models trained, tested, and saved successfully.\n")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_model = None

    if request.method == 'POST':
        selected_model = request.form['model']
        model = trained_models[selected_model]

        type_map = {'H': 1, 'M': 2, 'L': 3}
        type_val = type_map[request.form['type']]
        air_temp = float(request.form['air_temp'])
        process_temp = float(request.form['process_temp'])
        rot_speed = float(request.form['rot_speed'])
        torque = float(request.form['torque'])
        tool_wear = float(request.form['tool_wear'])

        power = rot_speed * torque
        temp_diff = process_temp - air_temp
        tool_torque = tool_wear * torque

        input_data = pd.DataFrame([{
            'Type': type_val,
            'Air temperature K': air_temp,
            'Process temperature K': process_temp,
            'Rotational speed rpm': rot_speed,
            'Torque Nm': torque,
            'Tool wear min': tool_wear,
            'Power': power,
            'Temp_diff': temp_diff,
            'Tool_Torque': tool_torque
        }])

        prediction = model.predict(input_data)[0]
        result = "Failure" if prediction == 1 else "No Failure"

    return render_template('index.html', result=result, selected_model=selected_model)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)