# RoadGuard AI — Streamlit application
# Colab setup:
# !pip -q install streamlit scikit-learn pandas numpy plotly
# Then save this file as app.py and run with a Streamlit-compatible tunnel.

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

st.set_page_config(page_title="RoadGuard AI", page_icon="🚦", layout="wide")

SEED = 42
rng = np.random.default_rng(SEED)

FEATURES = [
    "traffic_density", "avg_speed", "rainfall", "visibility",
    "road_wetness", "incident_history", "sensor_reliability"
]

@st.cache_data
def generate_demo_data(n=5000, seed=42):
    r = np.random.default_rng(seed)
    traffic = r.uniform(0, 100, n)
    speed = r.uniform(5, 100, n)
    rain = r.gamma(1.4, 3.0, n).clip(0, 40)
    visibility = r.uniform(0.2, 10, n)
    wetness = np.clip(r.normal(rain / 40, 0.15, n), 0, 1)
    history = r.poisson(1.2, n).clip(0, 8)
    reliability = r.uniform(0.45, 1.0, n)

    # Synthetic rule for demonstration only; not a real-world accident model.
    logits = (
        -3.0 + 0.035 * traffic
        + 0.065 * rain
        + 0.9 * wetness
        + 0.28 * history
        + 0.20 * np.maximum(speed - 70, 0) / 10
        + 0.35 * (3 - visibility).clip(0) 
    )
    probability = 1 / (1 + np.exp(-logits))
    labels = r.binomial(1, probability)
    df = pd.DataFrame({
        "traffic_density": traffic,
        "avg_speed": speed,
        "rainfall": rain,
        "visibility": visibility,
        "road_wetness": wetness,
        "incident_history": history,
        "sensor_reliability": reliability,
        "high_risk": labels
    })
    return df

@st.cache_resource
def train_model():
    data = generate_demo_data()
    X = data[FEATURES]
    y = data["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, (probs >= 0.5).astype(int)),
        "auc": roc_auc_score(y_test, probs),
        "y_test": y_test.to_numpy(),
        "probs": probs
    }
    return model, data, metrics

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

model, demo_data, metrics = train_model()

st.title("🚦 RoadGuard AI")
st.caption("Traffic congestion and road-safety risk intelligence — research prototype")
st.warning(
    "Risk scores, sample road locations, and traffic values use synthetic demo data, not "
    "live observations. Do not use these predictions for real driving, "
    "emergency response, or traffic-control decisions."
)

with st.sidebar:
    st.header("Scenario controls")
    st.caption("Adjust conditions to simulate a road segment.")
    traffic = st.slider("Traffic density (%)", 0, 100, 72)
    speed = st.slider("Average speed (km/h)", 5, 120, 65)
    rainfall = st.slider("Rainfall (mm/h)", 0.0, 40.0, 5.0, 0.5)
    visibility = st.slider("Visibility (km)", 0.2, 10.0, 5.0, 0.1)
    wetness = st.slider("Road wetness (0–1)", 0.0, 1.0, 0.25, 0.05)
    history = st.slider("Historical incident indicator (0–8)", 0, 8, 2)
    reliability = st.slider("Sensor reliability (0–1)", 0.0, 1.0, 0.85, 0.05)
    st.divider()
    st.subheader("Sensor failure simulation")
    missing_sensor = st.selectbox(
        "Simulate unreliable or missing input",
        ["None", "Traffic density", "Average speed", "Weather", "Visibility"]
    )
    st.caption("A missing input is replaced with a neutral training-data median.")

scenario = {
    "traffic_density": float(traffic),
    "avg_speed": float(speed),
    "rainfall": float(rainfall),
    "visibility": float(visibility),
    "road_wetness": float(wetness),
    "incident_history": float(history),
    "sensor_reliability": float(reliability)
}

feature_to_hide = {
    "Traffic density": "traffic_density",
    "Average speed": "avg_speed",
    "Weather": "rainfall",
    "Visibility": "visibility"
}.get(missing_sensor)

model_input = scenario.copy()
if feature_to_hide:
    model_input[feature_to_hide] = float(demo_data[feature_to_hide].median())
    model_input["sensor_reliability"] = min(model_input["sensor_reliability"], 0.45)

X_scenario = pd.DataFrame([model_input], columns=FEATURES)
risk_probability = float(model.predict_proba(X_scenario)[0, 1])

# Model disagreement proxy: variation across individual trees.
tree_probs = np.array([
    tree.predict_proba(X_scenario)[0, 1] for tree in model.estimators_
])
model_spread = float(np.std(tree_probs))
data_penalty = (1.0 - model_input["sensor_reliability"]) * 0.25
uncertainty = float(np.clip(model_spread + data_penalty, 0, 1))

# Decision threshold is illustrative and should be calibrated on real data.
if risk_probability >= 0.70:
    risk_label = "HIGH"
    recommendation = "Prioritize a safety review; consider a warning and verify conditions."
elif risk_probability >= 0.40:
    risk_label = "MODERATE"
    recommendation = "Monitor conditions and validate the available sensor observations."
else:
    risk_label = "LOW"
    recommendation = "Continue monitoring; reassess if conditions or data quality change."

if uncertainty >= 0.30 or model_input["sensor_reliability"] < 0.5:
    recommendation += " Confidence is limited: obtain a fresh or independent observation."

st.subheader("Current scenario")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimated risk score", f"{risk_probability * 100:.1f}/100")
c2.metric("Risk category", risk_label)
c3.metric("Uncertainty indicator", f"{uncertainty * 100:.1f}%")
c4.metric("Sensor reliability", f"{model_input['sensor_reliability'] * 100:.0f}%")

left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Risk probability")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d9534f"},
            "steps": [
                {"range": [0, 40], "color": "#d9ead3"},
                {"range": [40, 70], "color": "#fff2cc"},
                {"range": [70, 100], "color": "#f4cccc"}
            ],
            "threshold": {"line": {"color": "#333", "width": 3},
                          "thickness": 0.8, "value": 70}
        }
    ))
    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
    st.plotly_chart(gauge, use_container_width=True)

with right:
    st.markdown("### Recommended response")
    st.info(recommendation)
    st.markdown("**Inputs used by the model**")
    st.dataframe(
        pd.DataFrame({
            "Feature": list(model_input.keys()),
            "Value": [round(v, 3) for v in model_input.values()]
        }),
        use_container_width=True, hide_index=True
    )

st.divider()
st.subheader("Road-segment risk map")
st.caption(
    "Choose a city to view generated sample road locations. These traffic values and scores "
    "are synthetic, not live observations."
)

city_centers = {
    "Nagpur, India": (21.1458, 79.0882),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi, India": (28.6139, 77.2090),
    "Bengaluru, India": (12.9716, 77.5946),
    "Hyderabad, India": (17.3850, 78.4867),
    "Chennai, India": (13.0827, 80.2707),
    "Kolkata, India": (22.5726, 88.3639),
    "Pune, India": (18.5204, 73.8567),
    "Jaipur, India": (26.9124, 75.7873),
    "Ahmedabad, India": (23.0225, 72.5714),
    "Lucknow, India": (26.8467, 80.9462),
    "Bhopal, India": (23.2599, 77.4126),
    "New York, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Chicago, USA": (41.8781, -87.6298),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "London, UK": (51.5072, -0.1276),
    "Paris, France": (48.8566, 2.3522),
    "Berlin, Germany": (52.5200, 13.4050),
    "Rome, Italy": (41.9028, 12.4964),
    "Istanbul, Turkey": (41.0082, 28.9784),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "Dubai, UAE": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Jakarta, Indonesia": (-6.2088, 106.8456),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Sydney, Australia": (-33.8688, 151.2093),
    "São Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
}
location_options = ["Other location (enter coordinates)", *city_centers]
selected_location = st.selectbox(
    "Select a city for the demo map",
    location_options,
    index=location_options.index("Nagpur, India")
)

if selected_location == "Other location (enter coordinates)":
    coordinate_col1, coordinate_col2, coordinate_col3 = st.columns([2, 1, 1])
    with coordinate_col1:
        custom_location_name = st.text_input("Location name", value="Custom location")
    with coordinate_col2:
        center_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
    with coordinate_col3:
        center_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)
    map_location = custom_location_name.strip() or "Custom location"
else:
    center_lat, center_lon = city_centers[selected_location]
    map_location = selected_location

@st.cache_data
def generate_demo_roads(location_name, center_lat, center_lon, seed=7):
    stable_location_seed = sum(ord(character) for character in location_name)
    r = np.random.default_rng(seed + stable_location_seed)
    n = 30
    # Offset generated markers around the selected location for demo visualization only.
    lat = np.clip(center_lat + r.uniform(-0.06, 0.06, n), -90, 90)
    lon = np.clip(center_lon + r.uniform(-0.06, 0.06, n), -180, 180)
    traffic = r.uniform(10, 100, n)
    rain = r.uniform(0, 20, n)
    visibility = r.uniform(0.5, 10, n)
    wet = r.uniform(0, 1, n)
    incidents = r.integers(0, 6, n)
    rel = r.uniform(0.5, 1, n)
    roads = pd.DataFrame({
        "road_segment": [f"{location_name} Demo Segment {i+1:02d}" for i in range(n)],
        "location": location_name,
        "lat": lat, "lon": lon,
        "traffic_density": traffic,
        "avg_speed": r.uniform(10, 90, n),
        "rainfall": rain,
        "visibility": visibility,
        "road_wetness": wet,
        "incident_history": incidents,
        "sensor_reliability": rel
    })
    roads["risk_probability"] = model.predict_proba(roads[FEATURES])[:, 1]
    roads["risk_category"] = pd.cut(
        roads["risk_probability"], bins=[-0.001, 0.4, 0.7, 1.0],
        labels=["Low", "Moderate", "High"]
    ).astype(str)
    roads["traffic_condition"] = pd.cut(
        roads["traffic_density"],
        bins=[0, 33, 66, 100],
        labels=["Light", "Moderate", "Heavy"],
        include_lowest=True
    ).astype(str)
    return roads

roads = generate_demo_roads(map_location, center_lat, center_lon)
map_fig = px.scatter_map(
    roads, lat="lat", lon="lon", color="risk_category",
    size="traffic_density", hover_name="road_segment",
    hover_data={
        "risk_probability": ":.2f",
        "traffic_density": ":.1f",
        "sensor_reliability": ":.2f",
        "location": True,
        "lat": False, "lon": False
    },
    color_discrete_map={"Low": "#2ca25f", "Moderate": "#f0ad4e", "High": "#d9534f"},
    zoom=10, height=500,
    title=f"Synthetic road-risk examples — {map_location}"
)
map_fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(map_fig, use_container_width=True)

st.subheader(f"Demo traffic condition by location — {map_location}")
st.caption(
    "Each marker shows a demo location's traffic condition: Light (0–33%), "
    "Moderate (>33–66%), or Heavy (>66–100%). Traffic values are synthetic, not live data."
)
traffic_fig = px.scatter_map(
    roads, lat="lat", lon="lon", color="traffic_condition",
    size="traffic_density", hover_name="road_segment",
    hover_data={
        "traffic_density": ":.1f",
        "avg_speed": ":.1f",
        "risk_category": True,
        "location": True,
        "lat": False, "lon": False
    },
    color_discrete_map={
        "Light": "#2ca25f",
        "Moderate": "#f0ad4e",
        "Heavy": "#d9534f"
    },
    category_orders={"traffic_condition": ["Light", "Moderate", "Heavy"]},
    zoom=10, height=500,
    title=f"Synthetic traffic examples — {map_location}"
)
traffic_fig.update_layout(
    map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0),
    legend_title_text="Traffic condition"
)
st.plotly_chart(traffic_fig, use_container_width=True)

st.divider()
tab1, tab2, tab3 = st.tabs(["Model evaluation", "Feature importance", "Data & downloads"])

with tab1:
    st.markdown("### Holdout evaluation")
    st.caption("Metrics below measure performance on a held-out portion of the synthetic dataset only.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("ROC-AUC", f"{metrics['auc']:.3f}")
    m3.metric("Holdout rows", f"{len(metrics['y_test']):,}")
    eval_df = pd.DataFrame({
        "Actual label": metrics["y_test"],
        "Predicted probability": metrics["probs"]
    })
    st.dataframe(eval_df.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download holdout predictions",
        eval_df.to_csv(index=False).encode("utf-8"),
        file_name="roadguard_holdout_predictions.csv",
        mime="text/csv"
    )

with tab2:
    importances = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                 title="Random Forest feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Feature importance is descriptive, not causal evidence.")

with tab3:
    st.markdown("### Synthetic training data")
    st.dataframe(demo_data.head(100), use_container_width=True)
    st.download_button(
        "Download synthetic dataset",
        demo_data.to_csv(index=False).encode("utf-8"),
        file_name="roadguard_synthetic_training_data.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download demo road segments",
        roads.to_csv(index=False).encode("utf-8"),
        file_name="roadguard_demo_road_segments.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Research notes")
st.markdown(
    """
    - This application is a research prototype, not a certified road-safety product.
    - Synthetic labels are generated from an illustrative rule and cannot establish real-world validity.
    - The uncertainty indicator combines tree disagreement and a simple data-quality penalty; it is not a calibrated probability of error.
    - Before real-world use, validate against properly sourced, representative data and assess calibration, false alarms, missing-data robustness and performance across locations.
    - A patent filing requires a separate prior-art review of the specific technical mechanism and claims.
    """
)


st.divider()
st.subheader("🤖 Road Safety AI Chatbot — powered by Ollama")
st.caption("Ask about safe driving, traffic rules, bad weather, road hazards, or the current demo scenario. Replies are generated locally by Ollama.")

with st.sidebar:
    st.divider()
    st.subheader("Road Safety Chatbot")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.selectbox(
        "Ollama model",
        ["llama3.2", "qwen2.5", "mistral", "gemma2"],
        index=0,
        help="Choose a model that you have downloaded with `ollama pull`."
    )

if "roadguard_chat_history" not in st.session_state:
    st.session_state.roadguard_chat_history = []

for chat_message in st.session_state.roadguard_chat_history:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

user_question = st.chat_input("Ask Road Safety AI a question...")

if user_question:
    st.session_state.roadguard_chat_history.append(
        {"role": "user", "content": user_question}
    )
    with st.chat_message("user"):
        st.markdown(user_question)

    current_scenario = (
        f"Current demo scenario: estimated risk score {risk_probability * 100:.1f}/100; "
        f"risk category {risk_label}; uncertainty indicator {uncertainty * 100:.1f}%; "
        f"traffic density {traffic}%; average speed {speed} km/h; rainfall {rainfall} mm/h; "
        f"visibility {visibility} km; road wetness {wetness}; incident history {history}; "
        f"sensor reliability {model_input['sensor_reliability']:.2f}. "
        "These values come from a synthetic demonstration model, not live traffic data."
    )

    system_prompt = (
        "You are RoadGuard Road Safety AI, a helpful assistant focused on road safety, "
        "defensive driving, traffic congestion, weather hazards, pedestrian and cyclist safety, "
        "and general road-safety education. Give practical, clear, concise advice. "
        "Never claim to have live traffic, location, weather, police, or emergency-service data. "
        "Treat the supplied scenario as synthetic demo data, not a reliable real-world prediction. "
        "Do not encourage illegal or dangerous driving. For immediate danger or a crash, tell "
        "the user to move to a safe place if possible and contact local emergency services. "
        "Do not present yourself as a substitute for emergency responders or official road rules."
    )

    ollama_messages = [{"role": "system", "content": system_prompt}]
    ollama_messages.append({
        "role": "system",
        "content": current_scenario
    })
    for old_message in st.session_state.roadguard_chat_history[-12:]:
        ollama_messages.append({
            "role": old_message["role"],
            "content": old_message["content"]
        })

    with st.chat_message("assistant"):
        with st.spinner(f"Road Safety AI is thinking with {ollama_model}..."):
            try:
                response = requests.post(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": ollama_messages,
                        "stream": False
                    },
                    timeout=120
                )
                response.raise_for_status()
                answer = response.json().get("message", {}).get("content", "").strip()
                if not answer:
                    answer = "Ollama returned an empty response. Please try asking again."
                st.markdown(answer)
                st.session_state.roadguard_chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except requests.exceptions.ConnectionError:
                answer = (
                    "I couldn't connect to Ollama at "
                    f"`{ollama_url}`. Make sure Ollama is installed and running, then try again."
                )
                st.error(answer)
                st.session_state.roadguard_chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except requests.exceptions.Timeout:
                answer = "Ollama took too long to respond. Try again or select a smaller model."
                st.error(answer)
                st.session_state.roadguard_chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except requests.exceptions.HTTPError as exc:
                details = ""
                try:
                    details = response.json().get("error", "")
                except Exception:
                    pass
                answer = (
                    f"Ollama request failed ({response.status_code}). "
                    f"{details or 'Check that the selected model is downloaded.'} "
                    f"Run `ollama pull {ollama_model}` in PowerShell if needed."
                )
                st.error(answer)
                st.session_state.roadguard_chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                answer = f"Unexpected chatbot error: {exc}"
                st.error(answer)
                st.session_state.roadguard_chat_history.append(
                    {"role": "assistant", "content": answer}
                )

if st.session_state.roadguard_chat_history:
    if st.button("Clear chatbot conversation"):
        st.session_state.roadguard_chat_history = []
        st.rerun()
