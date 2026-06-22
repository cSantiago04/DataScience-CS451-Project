"""
streamlit_app.py — Global Earthquake Clustering
================================================
Run with:  streamlit run streamlit_app.py

Expects:
  models/rf_model.pkl
  models/dbscan_model.pkl
  data/features.pkl
"""

import pickle
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Earthquake Clustering",
    page_icon="🌍",
    layout="wide",
)

# ── model + data loading (cached — only runs once) ─────────────────────────
@st.cache_resource
def load_models():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/dbscan_model.pkl", "rb") as f:
        dbscan = pickle.load(f)
    return rf, dbscan

@st.cache_data
def load_data():
    with open("data/features.pkl", "rb") as f:
        data = pickle.load(f)
    return data["df"], data["feature_names"], data["scaler"]

rf_model, dbscan_model = load_models()
df, feature_names, scaler = load_data()

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Earthquake Explorer")
    st.markdown("---")
    page = st.radio("Navigate", ["Earthquake Explorer", "Significance Predictor"])
    st.markdown("---")
    st.markdown("**Dataset stats**")
    st.metric("Total events", f"{len(df):,}")
    st.metric("Date range", f"{df['year'].min():.0f} – {df['year'].max():.0f}")
    st.metric("Clusters found", f"{df['cluster_label'].nunique()}")
    st.markdown("---")
    st.caption("CS 451 — Final Project\nGlobal Earthquake Clustering")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EARTHQUAKE EXPLORER
# ══════════════════════════════════════════════════════════════════════════
if page == "Earthquake Explorer":
    st.title("Earthquake Explorer")
    st.markdown(
        "Use the filters below to explore global earthquake patterns. "
        "The heatmap weights events by magnitude — brighter = more energy released."
    )

    # ── filters ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        mag_range = st.slider(
            "Magnitude range",
            min_value=float(df["magnitude"].min()),
            max_value=float(df["magnitude"].max()),
            value=(5.0, float(df["magnitude"].max())),
            step=0.1,
        )
    with col2:
        depth_cats = st.multiselect(
            "Depth category",
            options=["shallow", "intermediate", "deep"],
            default=["shallow", "intermediate", "deep"],
        )
    with col3:
        year_range = st.slider(
            "Year range",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=(int(df["year"].min()), int(df["year"].max())),
        )

    # ── NEW: cluster filter ───────────────────────────────────────────────
    st.markdown("**Filter by cluster:**")
    all_clusters = sorted([c for c in df["cluster_label"].unique() if c != -1])
    
    cluster_cols = st.columns(6)
    selected_clusters = []
    for i, cluster_id in enumerate(all_clusters):
        with cluster_cols[i % 6]:
            if st.checkbox(f"Cluster {cluster_id}", value=True, key=f"cluster_{cluster_id}"):
                selected_clusters.append(cluster_id)
    
    # Option to show noise
    show_noise = st.checkbox("Show noise points", value=False)
    
    # ── filter dataframe ──────────────────────────────────────────────────
    filtered = df[
        (df["magnitude"] >= mag_range[0]) &
        (df["magnitude"] <= mag_range[1]) &
        (df["depth_category"].isin(depth_cats)) &
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
    ]
    
    # Apply cluster filter
    if selected_clusters:
        cluster_mask = filtered["cluster_label"].isin(selected_clusters)
        if show_noise:
            cluster_mask = cluster_mask | (filtered["cluster_label"] == -1)
        filtered = filtered[cluster_mask]
    elif show_noise:
        filtered = filtered[filtered["cluster_label"] == -1]
    else:
        filtered = filtered[filtered["cluster_label"].isin(selected_clusters)]

    # ── quick stats ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Events shown",    f"{len(filtered):,}")
    m2.metric("Avg magnitude",   f"{filtered['magnitude'].mean():.2f}" if len(filtered) > 0 else "N/A")
    m3.metric("Max magnitude",   f"{filtered['magnitude'].max():.1f}" if len(filtered) > 0 else "N/A")
    m4.metric("Avg depth (km)",  f"{filtered['depth'].mean():.0f}" if len(filtered) > 0 else "N/A")

    st.markdown("---")

    # ── folium heatmap with cluster colors ────────────────────────────────
    st.subheader("Global seismic map")

    if len(filtered) == 0:
        st.warning("No events match the current filters.")
    else:
        # Sample for performance
        sample = filtered if len(filtered) <= 20_000 else filtered.sample(20_000, random_state=42)

        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
        
        # Define cluster colors
        colors = ["red","blue","green","purple","orange","darkred","lightblue",
                  "darkblue","cadetblue","darkgreen","pink"]
        
        # Plot points colored by cluster
        for cluster_id in selected_clusters:
            cluster_data = sample[sample["cluster_label"] == cluster_id]
            if len(cluster_data) > 0:
                for _, row in cluster_data.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=2,
                        color=colors[cluster_id % len(colors)],
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"Cluster {cluster_id}<br>Mag: {row['magnitude']:.2f}",
                    ).add_to(m)
        
        # Plot noise if selected
        if show_noise:
            noise_data = sample[sample["cluster_label"] == -1]
            for _, row in noise_data.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=1,
                    color="gray",
                    fill=True,
                    fill_opacity=0.3,
                ).add_to(m)

        st_folium(m, width="100%", height=500)

    # ── cluster breakdown table ───────────────────────────────────────────
    if len(filtered) > 0 and "cluster_label" in filtered.columns:
        st.subheader("Cluster breakdown (filtered data)")
        cluster_summary = (
            filtered[filtered["cluster_label"] != -1]
            .groupby("cluster_label")
            .agg(
                count=("magnitude", "count"),
                avg_magnitude=("magnitude", "mean"),
                avg_depth=("depth", "mean"),
                avg_significance=("sig", "mean"),
            )
            .round(2)
            .reset_index()
            .rename(columns={
                "cluster_label":    "Cluster",
                "count":            "Events",
                "avg_magnitude":    "Avg magnitude",
                "avg_depth":        "Avg depth (km)",
                "avg_significance": "Avg significance",
            })
        )
        st.dataframe(cluster_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIGNIFICANCE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "Significance Predictor":
    st.title("Earthquake Significance Predictor")
    st.markdown(
        "Enter the parameters of an earthquake event and the model will predict "
        "whether it is likely to be **low**, **medium**, or **high** significance."
    )
    st.info(
        "Significance is a composite USGS score accounting for magnitude, "
        "estimated casualties, damage, and media attention.",
        icon="ℹ️",
    )

    st.markdown("---")
    left, right = st.columns([1, 1])

    # ── inputs ────────────────────────────────────────────────────────────
    with left:
        st.subheader("Event parameters")
        
        # Interactive map for location selection
        st.markdown("**Select location on map:**")
        location_map = folium.Map(location=[35, 139], zoom_start=2, tiles="OpenStreetMap")
        location_map.add_child(folium.LatLngPopup())
        
        map_data = st_folium(location_map, width=700, height=300)
        
        # Extract clicked coordinates
        if map_data and map_data.get("last_clicked"):
            latitude = map_data["last_clicked"]["lat"]
            longitude = map_data["last_clicked"]["lng"]
            st.success(f"Selected: {latitude:.2f}°, {longitude:.2f}°")
        else:
            latitude = 35.0
            longitude = 139.0
            st.info("Click on the map to select a location, or use default (Tokyo area)")
        
        st.markdown("---")
        
        magnitude = st.slider("Magnitude",         min_value=1.0, max_value=10.0, value=6.0, step=0.1)
        depth     = st.slider("Depth (km)",        min_value=0,   max_value=700,  value=30,  step=5)
        tsunami   = st.selectbox("Tsunami", options=["No", "Yes"], index=0)
        tsunami_val = 0 if tsunami == "No" else 1
        hour      = st.slider("Hour of day",       min_value=0,   max_value=23,   value=12,  step=1)
        month     = st.slider("Month",             min_value=1,   max_value=12,   value=6,   step=1)

        predict_btn = st.button("Predict significance", type="primary", use_container_width=True)

    # ── prediction ────────────────────────────────────────────────────────
    with right:
        st.subheader("Prediction")

        if predict_btn:
            # --- engineer same features used during training ---
            depth_cat_shallow      = 1 if depth < 70  else 0
            depth_cat_intermediate = 1 if 70 <= depth < 300 else 0
            depth_cat_deep         = 1 if depth >= 300 else 0
            magnitude_energy       = 10 ** (1.5 * magnitude)
            
            ring_of_fire = 1 if (
                (-60 <= latitude <= 70) and
                (longitude >= 120 or longitude <= -60)
            ) else 0
            
            local_event_density = df["local_event_density"].median()

            input_features = {
                "latitude":              latitude,
                "longitude":             longitude,
                "depth":                 depth,
                "magnitude":             magnitude,
                "tsunami":               tsunami_val,
                "hour":                  hour,
                "month":                 month,
                "magnitude_energy":      magnitude_energy,
                "ring_of_fire":          ring_of_fire,
                "local_event_density":   local_event_density,
                "depth_shallow":         depth_cat_shallow,
                "depth_intermediate":    depth_cat_intermediate,
                "depth_deep":            depth_cat_deep,
            }

            input_df  = pd.DataFrame([input_features])[feature_names]
            input_scaled = scaler.transform(input_df)

            prediction   = rf_model.predict(input_scaled)[0]
            probabilities = rf_model.predict_proba(input_scaled)[0]
            confidence   = probabilities.max()

            label_map  = {0: "High", 1: "Low", 2: "Medium"}
            label = label_map.get(prediction, str(prediction))

            st.metric("Predicted significance", label)
            st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")

            st.markdown("**Class probabilities**")
            prob_df = pd.DataFrame({
                "Class":       ["High", "Low", "Medium"],
                "Probability": [f"{p*100:.1f}%" for p in probabilities],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            notes = {
                "Low":    "This event is unlikely to cause significant damage or casualties.",
                "Medium": "This event may be felt widely and could cause minor damage.",
                "High":   "This event has potential for significant damage. Authorities should be alerted.",
            }
            st.info(notes.get(label, "Unknown significance level."))

            with st.expander("See engineered features sent to model"):
                st.json(input_features)

        else:
            st.markdown(
                "Select a location on the map, adjust the parameters, and click **Predict significance**."
            )
            st.markdown("\n**How it works:**")
            st.markdown("- Click the map to set latitude/longitude automatically")
            st.markdown("- Your inputs are feature-engineered the same way as training data")
            st.markdown("- The Random Forest model returns a class + confidence score")