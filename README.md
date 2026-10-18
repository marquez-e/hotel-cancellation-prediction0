# Hotel booking cancellation and overbooking optimizer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-orange.svg)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?logo=powerbi&logoColor=black)

An end-to-end data and machine learning solution designed to optimize strategic decisions in the hotel sector. This project minimizes the financial risk of empty rooms and the high costs of guest relocation by predicting booking cancellations and recommending optimal overbooking thresholds.

## Summary
Cancellation management and overbooking policies are critical challenges in the hospitality industry. To maximize revenue, hotels must find the perfect financial balance between unsold inventory and relocation penalties. 

This project tackles the problem through a complete data pipeline: 
1. **Robust ETL Process** that cleans data, prevents data leakage, and feeds a **PostgreSQL Star Schema**.
2. **Machine Learning Model (XGBoost)** optimized for AUC-PR to predict cancellation risk.
3. **Explainable AI (SHAP)** to identify the main drivers of cancellations.
4. **REST API (FastAPI)** to serve predictions in real-time interacting with a simulated Property Management System (PMS).
5. **Interactive Dashboard (Power BI)** integrating predictions and mathematical optimization (**Newsvendor Problem**) to recommend actionable overbooking limits.

## Architecture and tech stack

- **Data Engineering:** Python, Pandas, NumPy, SQLAlchemy, PostgreSQL (Star Schema)
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP (Explainability)
- **Backend & Serving:** FastAPI, Uvicorn
- **Data Visualization & BI:** Matplotlib, Seaborn, Power BI

## Data source
This project uses the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) by Jesse Mostipak. Download it and place the CSV files inside the /data folder before running the ETL.

## Repository structure

```
├── api/                  # FastAPI endpoints and API containerization
├── dwh/                  # SQL scripts for PostgreSQL Star Schema creation  
├── src/                  # Python scripts for ETL, model training and evaluation and PMS simulator
├── data/                 # Raw CSV file with data from Kaggle
├── notebooks/            # Jupyter notebooks for EDA, model evaluation, and SHAP analysis  
├── dashboard/            # Power BI file
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files and folders
└── README.md             # Project documentation  
```

## Key features

* **Advanced ETL & Data Engineering.** Handles missing values and treats frequent anomalies, ensures reproducibility and prevents data leakage by ordering datasets by arrival date and lead time, splits data chronologically (80% for training, 20% for future prediction simulation), and populates a pre-defined PostgreSQL Star Schema (Dimension and Fact tables) using optimized append operations.
* **High-Precision Predictions.** XGBoost model selected for its outstanding performance in highly imbalanced datasets (evaluated via AUC-PR).
* **Transparent AI.** SHAP values provide hotel managers with human-readable explanations of *why* a specific booking is likely to be canceled.
* **Smart Overbooking.** Translates predictions into business value by applying the Newsvendor Problem formula to calculate the exact number of rooms to overbook.

## Local setup & installation

**1. Clone the repository and install dependencies**
```bash
git clone https://github.com/your-username/hotel-overbooking-optimizer.git
cd hotel-overbooking-optimizer
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Database setup**
* Ensure PostgreSQL is installed and running.
* Create a new database named `hotel_dwh`.
* Run the SQL script located in `dwh/init_schema.sql` to generate the Star Schema structure with all Primary and Foreign Keys:
```bash
psql -U your_user -d hotel_dwh -f dwh/init_schema.sql
```

**3. Run the ETL pipeline**  
Process the raw data, train the model, and populate the PostgreSQL data warehouse (make sure your raw Kaggle datasets are placed inside the /data folder before running this step):  
```bash
python src/etl.py
```

**4. Start the services (requires 2 terminals)**
* Terminal 1: start the REST API
```bash
cd api
python main.py
```

* Terminal 2: start the PMS simulator
```bash
cd src
python pms_simulator.py
```

**5. What to expect now**  
Once both the API and the PMS Simulator are running, you will see the full end-to-end pipeline working together:
* **Terminal 1 (API):** Will log incoming `POST` requests in real-time as the simulator sends new booking data.
* **Terminal 2 (simulator):** Will display logs of the simulated future bookings being processed, showing the model's response (e.g., cancellation probabilities and SHAP reasons).
* **Database:** The `fact_future_simulation` table will be continuously updated with new rows, including the newly calculated `cancellation_probability`, `id_reason_1`, and `id_reason_2`.
* **Power BI:** Opening `dashboard/hotel_dashboard.pbix` and hitting "Refresh" will load the newly simulated predictions from PostgreSQL, updating the visuals and the overbooking recommendations dynamically.

## About
This project was developed as a Bachelor's Thesis in Computer Engineering, aiming to demonstrate a full-stack data science approach, bridging the gap between raw data, predictive modeling, and actionable business intelligence.
