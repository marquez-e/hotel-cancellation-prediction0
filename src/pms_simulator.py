import time
import json
import random
import requests
import pandas as pd
from sqlalchemy import create_engine

# Connection to Data Warehouse
url = 'postgresql://postgres:password@localhost:5432/hotel_bi'
engine = create_engine(url)

# FastAPI endpoint
url_api = 'http://127.0.0.1:8000/predict'

# PMS simulation function
def run_simulator():
    print("PMS Simulation. Connecting to API:", url_api)
    
    # Get the "future" reservations as new reservations
    df_reservations = pd.read_sql('''
                                SELECT 
                                    f.*,
                                    c.arrival_date, c.arrival_date_year, c.arrival_date_month, c.arrival_date_week_number,
                                    cl.country, cl.customer_type, cl.is_repeated_guest, cl.previous_cancellations, cl.previous_bookings_not_canceled,
                                    ch.market_segment, ch.distribution_channel, ch.agent, ch.company, ch.deposit_type,
                                    r.hotel, r.reserved_room_type, r.assigned_room_type, r.meal
                                FROM fact_future_simulation f
                                LEFT JOIN dim_calendar c ON f.id_calendar = c.id_calendar
                                LEFT JOIN dim_client cl ON f.id_client = cl.id_client
                                LEFT JOIN dim_channel ch ON f.id_channel = ch.id_channel
                                LEFT JOIN dim_room r ON f.id_room = r.id_room
                                WHERE f.cancellation_probability IS NULL
                                ''', engine)
    print(f"{len(df_reservations)} new reservations were loaded.\n")

    # Loop to simulate new reservations entering separately
    for index, row in df_reservations.iterrows():
        # Drop unnecessary columns like in ETL and convert to dict for API (JSON)
        reservation = row.drop(['is_canceled', 'arrival_date', 'cancellation_probability'], errors='ignore').to_dict()

        print(f"\nSending to API to evaluate the cancellation risk")

        response = requests.post(url_api, json=reservation)
            
        if response.status_code == 200:
                json = response.json()
                
                # Meaning of response
                probability = json.get('cancellation_probability', 0)
                alert = "High cancellation risk" if probability > 0.60 else ("Medium cancellation risk" if probability > 0.40 else "Low cancellation risk")
                print(f"API: {alert} ({probability:.1%})")
        else:
                print(f"Error {response.status_code}: {response.text}")
                    
        # Pause between reservations
        #time.sleep(random.uniform(4, 10))

if __name__ == "__main__":
    run_simulator()