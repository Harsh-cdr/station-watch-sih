"""
Station Watch — model training and SHAP explanation utilities.

The project detects anomalous AWS sensor readings using an Isolation Forest
and explains anomaly flags with SHAP.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

RANDOM_SEED = 42
FEATURES = ["temperature", "pressure", "humidity"]
MODEL_FILENAME = "isolation_forest_edge_model.pkl"


def generate_normal_data(n_samples: int = 2000) -> pd.DataFrame:
    """Create deterministic synthetic historical readings representing normal operation."""
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")

    # Use a fresh RandomState so every call produces the same reference
    # distribution. This keeps model training and confidence scaling reproducible.
    rng = np.random.RandomState(RANDOM_SEED)

    temperature = rng.normal(loc=25, scale=3, size=n_samples)
    pressure = rng.normal(loc=1013, scale=5, size=n_samples)
    humidity = rng.normal(loc=60, scale=10, size=n_samples)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
        },
        columns=FEATURES,
    )


def train_model(
    normal_df: pd.DataFrame,
    contamination: float = 0.03,
) -> IsolationForest:
    """Train an Isolation Forest using only normal historical readings."""
    missing = [feature for feature in FEATURES if feature not in normal_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_SEED,
    )
    model.fit(normal_df[FEATURES])
    return model


def get_demo_readings() -> pd.DataFrame:
    """Return fixed readings that reliably demonstrate normal and anomalous cases."""
    return pd.DataFrame(
        [
            {"temperature": 24.5, "pressure": 1012.0, "humidity": 58.0},
            {"temperature": 25.8, "pressure": 1010.5, "humidity": 61.0},
            {"temperature": 63.0, "pressure": 1011.0, "humidity": 59.0},
            {"temperature": 25.1, "pressure": 905.0, "humidity": 60.0},
            {"temperature": 24.9, "pressure": 1013.5, "humidity": 0.0},
        ],
        columns=FEATURES,
    )


def compute_confidence(
    score: float,
    threshold: float,
    score_min: float,
    score_max: float,
) -> float:
    """
    Convert the Isolation Forest decision score distance from the threshold
    into a 0–100 display value.

    This is a demo-oriented relative score, not a calibrated probability.
    """
    if score < threshold:
        distance = threshold - score
        max_distance = max(threshold - score_min, 1e-6)
    else:
        distance = score - threshold
        max_distance = max(score_max - threshold, 1e-6)

    return float(min(distance / max_distance, 1.0) * 100)


def render_confidence_bar(confidence: float, bar_length: int = 20) -> str:
    """Render a terminal confidence bar."""
    confidence = max(0.0, min(100.0, confidence))
    filled = int(round((confidence / 100) * bar_length))
    return f"[{'█' * filled}{'░' * (bar_length - filled)}] {confidence:.0f}%"


def explain_predictions(
    model: IsolationForest,
    readings: pd.DataFrame,
    score_threshold: float = 0.05,
    live_feed_delay: float = 1.5,
) -> None:
    """Run the scripted demo in the terminal with SHAP explanations."""
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(readings[FEATURES]))
    scores = model.decision_function(readings[FEATURES])

    training_data = generate_normal_data(n_samples=2000)
    training_scores = model.decision_function(training_data)
    score_min = float(training_scores.min())
    score_max = float(training_scores.max())

    print("\n--- Live sensor feed (simulated) ---\n")

    for i, reading in readings.iterrows():
        status = "ANOMALY" if scores[i] < score_threshold else "normal"
        confidence = compute_confidence(
            float(scores[i]),
            score_threshold,
            score_min,
            score_max,
        )

        print(f"[Live feed] Reading {i + 1} received...")
        if live_feed_delay > 0:
            time.sleep(live_feed_delay)

        print(
            f"   temp={reading.temperature:.1f}  "
            f"pressure={reading.pressure:.1f}  "
            f"humidity={reading.humidity:.1f}  "
            f"-> {status} (score={scores[i]:.3f})"
        )

        label = "Anomaly confidence" if status == "ANOMALY" else "Normal confidence"
        print(f"   {label}: {render_confidence_bar(confidence)}")

        if status == "ANOMALY":
            contributions = sorted(
                zip(FEATURES, shap_values[i]),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            top_feature, top_value = contributions[0]
            print(
                f"   -> Main driver: {top_feature} "
                f"(SHAP contribution={float(top_value):.3f})"
            )

        print()


def save_model(
    path: str | Path = MODEL_FILENAME,
    contamination: float = 0.03,
) -> Path:
    """Train and save the deterministic demo model."""
    output_path = Path(path)
    model = train_model(
        generate_normal_data(n_samples=2000),
        contamination=contamination,
    )
    joblib.dump(model, output_path)
    return output_path


if __name__ == "__main__":
    print("Training model on simulated historical 'normal' data...")
    output = save_model()
    print(f"Model saved to {output.resolve()}")

    model = joblib.load(output)
    explain_predictions(model, get_demo_readings())
