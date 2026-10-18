from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
import shap
from datetime import datetime

app = FastAPI(title="Cancellation probability prediction API")

# Loading the trained model
print("Loading model and SHAP explainer")
model_data = joblib.load('final_model.pkl')
model = model_data['model']
columns = model_data['columns']

# Connection to Data Warehouse
url = 'postgresql://postgres:password@localhost:5432/hotel_bi'
engine = create_engine(url)

# Set expected columns to come from PMS simulator
class ReservationInput(BaseModel):
    # IDs from foreign keys
    id_reservation: int
    id_calendar: int
    id_client: int
    id_channel: int
    id_room: int
    
    # fact_future_simulation
    lead_time: int
    adr: float
    total_guests: int
    total_nights: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    days_in_waiting_list: int
    booking_changes: int
    total_of_special_requests: int
    required_car_parking_spaces: int
    
    # Dim_calendar
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    
    # Dim_client
    country: str
    customer_type: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    
    # Dim_channel
    market_segment: str
    distribution_channel: str
    agent: int
    company: int
    deposit_type: str
    
    # Dim_room
    hotel: str
    reserved_room_type: str
    assigned_room_type: str
    meal: str

# Post request to predict cancellation probability and explain it
@app.post("/predict")
def predict(reservation: ReservationInput):
    # Convert JSON to DataFrame
    data = reservation.dict()
    df_input = pd.DataFrame([data])
    df_input = df_input.drop(columns=['id_reservation', 'id_calendar', 'id_client', 'id_room', 'id_channel'], errors='ignore')
    
    # Text variables will be encoded
    df_input = pd.get_dummies(df_input)

    # Dataframe hast to match the columns used in training
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Debug
    # print(f"\nDebugging. Columns active (value1):")
    # columns_act = [column for column in df_input.columns if df_input[column].iloc[0] > 0]
    # print(columns_act)
    
    # Predict
    probability = float(model.predict_proba(df_input)[0, 1])
    
    # Get reasons for the prediction using SHAP values
    # Inicialize SHAP explainer and calculate SHAP values for the input
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # Take class 1 (cancellation)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    print(f"Expected value: {explainer.expected_value}")
    print(f"SHAP values shape: {shap_values.shape if not isinstance(shap_values, list) else [s.shape for s in shap_values]}")
  
    # Get more important features for the prediction (DESC)
    feature_importance = pd.DataFrame({
        'feature': columns,
        'shap_value': shap_values[0],
        'feature_value': df_input.values[0]
    }).sort_values('shap_value', ascending=False, key=abs)
    
    # Top 2 most important factors
    first_factor = feature_importance.iloc[0]['feature'] if len(feature_importance) > 0 else "Unknown"
    second_factor = feature_importance.iloc[1]['feature'] if len(feature_importance) > 1 else "Unknown"
    
    # Loading prediction and reasons in database
    try:
        id_reservation = data.get('id_reservation')
        if id_reservation:
            with engine.begin() as conn:
                # Look for ids in the dimension table for reasons
                query_dim_reasons = text("SELECT name_reason, id_reason FROM dim_reason")
                df_reasons = pd.read_sql(query_dim_reasons, conn)

                # Convert into dictionary
                reasons_dict = dict(zip(df_reasons.name_reason, df_reasons.id_reason))

                # Look for Ids
                id_other_reasons = 999
                id_reason_1 = reasons_dict.get(first_factor, id_other_reasons)
                id_reason_2 = reasons_dict.get(second_factor, id_other_reasons)

                # Update the fact_future_simulation table with the new probability and reasons
                query = text("""
                                UPDATE fact_future_simulation 
                                SET cancellation_probability = :probability, id_reason_1 = :first_factor, id_reason_2 = :second_factor
                                WHERE id_reservation = :id
                            """)
                conn.execute(query, {"probability": probability, "first_factor": int(id_reason_1), "second_factor": int(id_reason_2), "id": id_reservation})
            print(f"Cancellation probability for {id_reservation} reservation: {probability}")
            # DEBUG. Show the reasons
            # print(f"Reasons: 1. {first_factor} (id: {id_reason_1}), 2. {second_factor} (id: {id_reason_2})")
        else:
            print("Error. The id_reservation wasnt received")
    except Exception as e:
        print(f"Error storing in Database: {e}")
    
    probability_float = float(probability)

    # Send response to simulator
    return {
        "status": "completed",
        "cancellation_probability": probability_float,
        "id_reason_1": first_factor,
        "id_reason_2": second_factor
    }