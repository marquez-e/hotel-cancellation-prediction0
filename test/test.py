import requests
import time

# FastAPI endpoint
url_api = 'http://127.0.0.1:8000/predict'

print("Starting test of the API")

# Testing when data sent is correct
data = {"id_reservation": 1, "id_calendar": 987, "id_client": 987, "id_channel": 987, "id_room": 987,
        "lead_time": 1, "adr": 50, "total_guests": 1, "total_nights": 2, "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2, "days_in_waiting_list": 0, "booking_changes": 0, "total_of_special_requests": 0,
        "required_car_parking_spaces": 1, "arrival_date_year": 2026, "arrival_date_month": "June", "arrival_date_week_number": 24,
        "country": "Spain", "customer_type": "Transient", "is_repeated_guest": 0, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "market_segment": "Online TA", "distribution_channel": "TA/TO",
        "agent": 0, "company": 0, "deposit_type": "No Deposit", "hotel": "City Hotel",
        "reserved_room_type": "A", "assigned_room_type": "A", "meal": "BB"}
start = time.time()
response = requests.post(url_api, json=data)
response_time = (time.time() - start) * 1000
print("\n·Testing API with correct reservation data:")
print(f"Test 1. Code {response.status_code}. {response.json().get('state', 'Ok')}")
print(f"Test 2. Latency: {response_time: .2f} ms.")

# Testing that response contains cancellation probability
print("\n·Testing if API response has cancellation probability (also with correct data):")
if 'cancellation_probability' in response.json():
    probability = response.json()['cancellation_probability']
    print(f"Test 3. API response contains cancellation probability ({probability}).")
else:
    print("Test 3. Error. API response doesn't contain cancellation probability.")

# Testing when data has missing values (ids)
data_missing = {"id_reservation": 987654321, "lead_time": 1, "adr": 50, "total_guests": 1, "total_nights": 2, "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2, "days_in_waiting_list": 0, "booking_changes": 0, "total_of_special_requests": 0,
        "required_car_parking_spaces": 1, "arrival_date_year": 2026, "arrival_date_month": "June", "arrival_date_week_number": 24,
        "country": "Spain", "customer_type": "Transient", "is_repeated_guest": 0, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "market_segment": "Online TA", "distribution_channel": "TA/TO",
        "agent": 0, "company": 0, "deposit_type": "No Deposit", "hotel": "City Hotel",
        "reserved_room_type": "A", "assigned_room_type": "A", "meal": "BB"}
response_missing = requests.post(url_api, json=data_missing)
print("\n·Testing API with missing data:")
print(f"Test 4. Code {response_missing.status_code}. {response_missing.json().get('state', 'Ok')}")
if (response_missing.status_code == 422):
    print("API correctly identifies missing data.")

# Testing when data sent has incorrect format
data_bad = {"id_reservation": "identification", "lead_time": "time", "adr": "adr", "total_guests": 1, "total_nights": 2, "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2, "days_in_waiting_list": 0, "booking_changes": 0, "total_of_special_requests": 0,
        "required_car_parking_spaces": 1, "arrival_date_year": 2026, "arrival_date_month": "June", "arrival_date_week_number": 24,
        "country": "Spain", "customer_type": "Transient", "is_repeated_guest": 0, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "market_segment": "Online TA", "distribution_channel": "TA/TO",
        "agent": 0, "company": 0, "deposit_type": "No Deposit", "hotel": "City Hotel",
        "reserved_room_type": "A", "assigned_room_type": "A", "meal": "BB"}
response2 = requests.post(url_api, json=data_bad)
print("\n·Testing API with wrong data formats:")
print(f"Test 5. Code {response2.status_code}. {response2.json().get('state', 'Ok')}")
if (response2.status_code == 422):
    print("API correctly identifies wrong data formats.")