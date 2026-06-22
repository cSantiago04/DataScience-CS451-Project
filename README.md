# Global Earthquake Data
**CS 451 — Final Project**

## Christian Meraz-Santiago
# Email: cjmerazsantiago@crimson.ua.edu

## Overview
End-to-end data science pipeline applying unsupervised and supervised machine learning to 33 years of global earthquake data (1990–2023).

## Models
1. **DBSCAN Clustering** — discovers 11 natural seismic zones
2. **Random Forest Classifier** — predicts earthquake significance (low/medium/high)

## Dataset
- Source: [Kaggle — All Earthquakes 1990–2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)
- Size: 543,500 events
- Features: 14 engineered features from magnitude, depth, location, and temporal data

## How to Run

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the notebook
Open `earthquake_clustering.ipynb` in Jupyter and run all cells to train models.

### 3. Launch Streamlit demo
```bash
python -m streamlit run streamlit_app.py
```

## Results
- **DBSCAN**: Silhouette 0.50, 11 clusters, 1% noise
- **Random Forest**: 100% accuracy (magnitude is highly predictive of significance)

## Acknowledgments
Dataset provided by Alessandro Lobello via Kaggle. USGS Earthquake Hazards Program for the original data collection.