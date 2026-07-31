from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load pre-trained models once at startup (fast, low memory).
# Do NOT retrain here on every boot -- that was the old behavior and it
# will make deployments slow/crash-prone on hosts with limited build time.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

trained_models = {
    'decision_tree': joblib.load(os.path.join(MODEL_DIR, 'decision_tree_model.pkl')),
    'random_forest': joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl')),
    'xgboost': joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl')),
    'lightgbm': joblib.load(os.path.join(MODEL_DIR, 'lightgbm_model.pkl')),
}

FEATURE_COLUMNS = [
    'Type', 'Air temperature K', 'Process temperature K',
    'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
    'Power', 'Temp_diff', 'Tool_Torque'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_model = None

    if request.method == 'POST':
        try:
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
            }])[FEATURE_COLUMNS]

            prediction = model.predict(input_data)[0]
            result = "Failure" if prediction == 1 else "No Failure"
        except (KeyError, ValueError):
            result = "Invalid input"

    return render_template('index.html', result=result, selected_model=selected_model)


@app.route('/health')
def health():
    return {"status": "ok"}, 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
