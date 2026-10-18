import os
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import numpy as np

# Connection to Data Warehouse
url = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(url)

# ETL Process
def run_etl():
    print("\nStarting ETL process.")
    
    # ETL Phase 1: extraction
    data_path = Path(__file__).parent.parent / 'data' / 'hotel_bookings.csv'
    df = pd.read_csv(data_path)
    len_before = len(df)
    
    # ETL Phase 2: transformation
    print("\nTransforming data.")

    # Fill missing values or undefined values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    df['meal'] = df['meal'].replace({'Undefined': 'SC'})

    # Creation of new variables
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_date_month'] + '-' + 
        df['arrival_date_day_of_month'].astype(str)
    )
    
    # Treatment of frequent anomalies (and calculation of deleted rows for each anomaly)
    df = df[df['total_guests'] > 0]
    len_first_deletion = len(df)
    print(f"\n · {len_before - len_first_deletion} rows with 0 guests have been deleted.")

    df = df[df['total_nights'] > 0]
    len_second_deletion = len(df)
    print(f"\n · {len_first_deletion - len_second_deletion} rows with 0 nights in booking have been deleted.")

    df = df[df['adr'] >= 0]
    len_end = len(df)
    print(f"\n · {len_second_deletion - len_end} rows with negative ADR values have been deleted.")

    limit_adr = df['adr'].quantile(0.99)
    df['adr'] = np.where(df['adr'] > limit_adr, limit_adr, df['adr'])
    
    limit_lead = df['lead_time'].quantile(0.99)
    df['lead_time'] = np.where(df['lead_time'] > limit_lead, limit_lead, df['lead_time'])

    print(f"\nClean-up completed: \nDataset contained {len_before} rows before cleaning, and {len_end} rows after cleaning. \n{len_before - len_end} anomalous rows have been removed.")
    
    # While development, truncate tables to avoid duplicates
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE dim_calendar RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE dim_client RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE dim_channel RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE dim_room RESTART IDENTITY  CASCADE"))
        conn.execute(text("TRUNCATE TABLE fact_reservations RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE fact_future_simulation RESTART IDENTITY CASCADE"))
        conn.commit()

    # Ordering dataset by arrival date for future data cut, by lead_time for better reproducibility and avoid data leakage (future reservations with past lead time)
    df = df.sort_values(by=['arrival_date', 'lead_time']).reset_index(drop=True)
    
    # ETL Phase 3: loading data in PostgreSQL (in fact and dimension tables)
    # Tables created in PostgreSQL to set relationships and constraints, that's why if_exists='append'

    # 3.1. Dimension tables
    # First get only calendar/client/channel/room-related columns and drop duplicates.
    # Then store in respective dimension table from database. Then read table again to finally merge and get the id (needed for fact tables)
    print("\nFilling dimension tables.")

    # Dimension calendar
    dim_calendar = df[['arrival_date', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number']].drop_duplicates()
    dim_calendar.to_sql('dim_calendar', engine, if_exists='append', index=False)
    dim_calendar_postgres = pd.read_sql("SELECT * FROM dim_calendar", engine, parse_dates=['arrival_date'])
    df = df.merge(dim_calendar_postgres, on=['arrival_date', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number'], how='left')

    # Dimension client
    dim_client = df[['country', 'customer_type', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled']].drop_duplicates()
    dim_client.to_sql('dim_client', engine, if_exists='append', index=False)
    dim_client_postgres = pd.read_sql("SELECT * FROM dim_client", engine)
    df = df.merge(dim_client_postgres, on=['country', 'customer_type', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled'], how='left')
   
    # Dimension distribution channel
    dim_channel = df[['market_segment', 'distribution_channel', 'agent', 'company', 'deposit_type']].drop_duplicates()
    dim_channel.to_sql('dim_channel', engine, if_exists='append', index=False)
    dim_channel_postgres = pd.read_sql("SELECT * FROM dim_channel", engine)
    df = df.merge(dim_channel_postgres, on=['market_segment', 'distribution_channel', 'agent', 'company', 'deposit_type'], how='left')

    # Dimension hotel room
    dim_room = df[['hotel', 'reserved_room_type', 'assigned_room_type', 'meal']].drop_duplicates()
    dim_room.to_sql('dim_room', engine, if_exists='append', index=False)
    dim_room_postgres = pd.read_sql("SELECT * FROM dim_room", engine)
    df = df.merge(dim_room_postgres, on=['hotel', 'reserved_room_type', 'assigned_room_type', 'meal'], how='left')

    print("Dimension tables filled.")

    # 3.2. Fact table
    columns_fact_table = [
        'id_calendar', 'id_client', 'id_channel', 'id_room', 
        'is_canceled', 'lead_time', 'adr', 'total_guests', 'total_nights',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'days_in_waiting_list',
        'booking_changes', 'total_of_special_requests','required_car_parking_spaces'
    ]
    df_facts = df[columns_fact_table].copy()

    # Add empty new columns for predictions
    df_facts['cancellation_probability'] = None
    df_facts['id_reason_1'] = None
    df_facts['id_reason_2'] = None

    # Cutting data for future predictions (80/20)
    # Getting the index that corresponds to the 80% of the data (oldest data)
    print("\nCutting dataset chronologically for future predictions.")
    cut_index = int(0.8 * len(df))

    # Get the 80% of the data for training and 20% for simulating future predictions
    df_past = df_facts.iloc[:cut_index]
    df_future = df_facts.iloc[cut_index:]

    # 3.3. Load final fact tables in database
    print("Storing past data in 'fact_reservations' and future data in 'fact_future_simulation'.")
    df_past.to_sql('fact_reservations', engine, if_exists='append', index=False)
    df_future.to_sql('fact_future_simulation', engine, if_exists='append', index=False)
    
    print(f"\nETL process completed successfully. {len(df_past)} rows for past reservatinos and {len(df_future)} rows for fuuture reservations.")

if __name__ == "__main__":
    run_etl()