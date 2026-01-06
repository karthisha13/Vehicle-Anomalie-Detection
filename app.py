from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    total = anomaly = None
    sensor_issues = {}
    top_sensor = severity = recommendation = None

    if request.method == "POST":
        file = request.files["dataset"]

        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            features = [
                'rpm_residual',
                'temp_residual',
                'load_residual',
                'voltage_residual',
                'missing_can_frames',
                'can_interval_std'
            ]

            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=200,
                contamination=0.15,
                random_state=42
            )
            model.fit(X_scaled)

            df['anomaly'] = model.predict(X_scaled)

            total = len(df)
            anomalies = df[df['anomaly'] == -1]
            anomaly = len(anomalies)

            if anomaly > 0:
                sensor_issues = {
                    "Engine RPM Sensor": anomalies['rpm_residual'].abs().mean(),
                    "Engine Temperature Sensor": anomalies['temp_residual'].abs().mean(),
                    "Engine Load Sensor": anomalies['load_residual'].abs().mean(),
                    "Battery Voltage Sensor": anomalies['voltage_residual'].abs().mean()
                }

                sensor_issues = dict(
                    sorted(sensor_issues.items(), key=lambda x: x[1], reverse=True)
                )

                top_sensor = list(sensor_issues.keys())[0]
                top_value = sensor_issues[top_sensor]

                if top_value > 15:
                    severity = "HIGH"
                    recommendation = "Immediate inspection required. Vehicle operation may be unsafe."
                elif top_value > 8:
                    severity = "MEDIUM"
                    recommendation = "Schedule maintenance soon to avoid system failure."
                else:
                    severity = "LOW"
                    recommendation = "Monitor sensor behavior during next vehicle cycle."

    return render_template(
        "index.html",
        total=total,
        anomaly=anomaly,
        sensor_issues=sensor_issues,
        top_sensor=top_sensor,
        severity=severity,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)
