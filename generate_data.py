import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sparkov_synthetic_data(num_rows, output_path):
    print(f"Generating {num_rows} rows of synthetic credit card transaction data...")
    
    np.random.seed(42)
    
    # Lists of sample data to mimic Sparkov simulation
    categories = [
        "personal_care", "shopping_net", "entertainment", "gas_transport", 
        "grocery_pos", "shopping_pos", "kids_pets", "home", "food_dining", 
        "health_fitness", "travel", "misc_pos", "grocery_net", "misc_net"
    ]
    
    genders = ["M", "F"]
    
    states = ["NY", "CA", "TX", "FL", "IL", "PA", "OH", "MI", "GA", "NC", "NJ", "VA", "WA", "AZ", "CO"]
    
    merchants = [
        "fraud_Rempel, Gohan and Kertzmann", "fraud_Heller, Moore and Reinger",
        "fraud_Kunde-Reinger", "fraud_Nader-Gaylord", "fraud_Bauch-Goldner",
        "fraud_Kuhn-Goldner", "fraud_Reinger Group", "fraud_Emard-Ullrich",
        "fraud_Cruickshank-Mills", "fraud_Adams-Frazier", "fraud_Kling-Simonis",
        "fraud_Ledner-Goodwin", "fraud_Glover-Hansen", "fraud_Stanton, Kihn and Kuhlman",
        "fraud_Haley, Jewess and Bechtelar", "fraud_Bernhard Inc", "fraud_Kozey-McGlynn"
    ]
    
    first_names = ["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Elizabeth", "David", "Barbara"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
    
    cities = [
        ("New York", "NY", 8336817), ("Los Angeles", "CA", 3979576), ("Chicago", "IL", 2693012),
        ("Houston", "TX", 2320268), ("Phoenix", "AZ", 1680992), ("Philadelphia", "PA", 1584064),
        ("San Antonio", "TX", 1547253), ("San Diego", "CA", 1423851), ("Dallas", "TX", 1343573),
        ("San Jose", "CA", 1021795), ("Austin", "TX", 978908), ("Jacksonville", "FL", 911507)
    ]
    
    jobs = ["Engineer", "Teacher", "Doctor", "Artist", "Nurse", "Manager", "Accountant", "Developer", "Salesperson", "Writer"]
    
    # Initialize dictionary for dataframe
    data = {
        "trans_date_trans_time": [],
        "cc_num": [],
        "merchant": [],
        "category": [],
        "amt": [],
        "first": [],
        "last": [],
        "gender": [],
        "street": [],
        "city": [],
        "state": [],
        "zip": [],
        "lat": [],
        "long": [],
        "city_pop": [],
        "job": [],
        "dob": [],
        "trans_num": [],
        "unix_time": [],
        "merch_lat": [],
        "merch_long": [],
        "is_fraud": []
    }
    
    start_date = datetime(2019, 1, 1)
    
    # Generate rows
    for i in range(num_rows):
        # Fraud probability (imbalanced class: ~0.5% - 1% fraud rate)
        # Let's make it 1.2% for better representation in small datasets
        is_fraud = 1 if np.random.rand() < 0.012 else 0
        
        # Date & Time
        random_seconds = np.random.randint(0, 365 * 2 * 24 * 3600)  # 2 years span
        trans_time = start_date + timedelta(seconds=random_seconds)
        
        # Injects fraud patterns:
        # 1. Late night transactions are more likely to be fraud
        # 2. Large transaction amounts are more likely to be fraud
        # 3. High risk categories are more likely to be fraud
        if is_fraud:
            # Shift trans_time to late night (11 PM - 4 AM) with 60% probability
            if np.random.rand() < 0.6:
                hour = np.random.choice([23, 0, 1, 2, 3, 4])
                trans_time = trans_time.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
            
            # Amount: fraud transactions are larger, but some are small test charges
            if np.random.rand() < 0.15:
                amt = np.random.uniform(1.00, 15.00)
            else:
                amt = np.random.uniform(150.0, 1200.0)
            
            # Categories: fraud matches high risk categories
            category = np.random.choice(["shopping_net", "grocery_net", "travel", "misc_net", "shopping_pos"])
        else:
            # Legitimate transactions:
            # Amount: mostly small, but some large
            rand_val = np.random.rand()
            if rand_val < 0.85:
                amt = np.random.exponential(scale=35.0) + 1.0  # highly skewed low amounts
            elif rand_val < 0.98:
                amt = np.random.uniform(50.0, 200.0)
            else:
                amt = np.random.uniform(200.0, 1500.0)  # large legitimate purchase
                
            category = np.random.choice(categories)
            
        trans_date_trans_time_str = trans_time.strftime("%Y-%m-%d %H:%M:%S")
        unix_time = int(trans_time.timestamp())
        
        # Demographics
        gender = np.random.choice(genders)
        cc_num = np.random.randint(1000000000000000, 9999999999999999, dtype=np.int64)
        merchant = np.random.choice(merchants)
        first = np.random.choice(first_names)
        last = np.random.choice(last_names)
        street = f"{np.random.randint(100, 9999)} {np.random.choice(last_names)} St"
        
        city_info = cities[np.random.randint(0, len(cities))]
        city, state, city_pop = city_info
        zip_code = np.random.randint(10000, 99999)
        
        # Latitude & Longitude (center around realistic US coords)
        base_lat = np.random.uniform(25.0, 48.0)
        base_long = np.random.uniform(-120.0, -70.0)
        
        # Distance: Fraud has larger distances on average, but overlaps exist
        if is_fraud:
            if np.random.rand() < 0.7:
                # remote/suspicious distance
                lat_diff = np.random.uniform(0.5, 3.5) * np.random.choice([-1, 1])
                long_diff = np.random.uniform(0.5, 3.5) * np.random.choice([-1, 1])
            else:
                # local fraud
                lat_diff = np.random.uniform(0.01, 0.2) * np.random.choice([-1, 1])
                long_diff = np.random.uniform(0.01, 0.2) * np.random.choice([-1, 1])
        else:
            if np.random.rand() < 0.95:
                # local transaction
                lat_diff = np.random.uniform(0.01, 0.2) * np.random.choice([-1, 1])
                long_diff = np.random.uniform(0.01, 0.2) * np.random.choice([-1, 1])
            else:
                # customer traveling or online delivery
                lat_diff = np.random.uniform(0.2, 3.5) * np.random.choice([-1, 1])
                long_diff = np.random.uniform(0.2, 3.5) * np.random.choice([-1, 1])
            
        lat = base_lat
        long = base_long
        merch_lat = base_lat + lat_diff
        merch_long = base_long + long_diff
        
        job = np.random.choice(jobs)
        
        # DOB (age range: 18 - 85)
        age = np.random.randint(18, 85)
        dob = (trans_time - timedelta(days=int(age * 365.25))).strftime("%Y-%m-%d")
        
        trans_num = f"{np.random.randint(100000000000, 999999999999, dtype=np.int64):x}"
        
        # Append data
        data["trans_date_trans_time"].append(trans_date_trans_time_str)
        data["cc_num"].append(cc_num)
        data["merchant"].append(merchant)
        data["category"].append(category)
        data["amt"].append(round(amt, 2))
        data["first"].append(first)
        data["last"].append(last)
        data["gender"].append(gender)
        data["street"].append(street)
        data["city"].append(city)
        data["state"].append(state)
        data["zip"].append(zip_code)
        data["lat"].append(round(lat, 4))
        data["long"].append(round(long, 4))
        data["city_pop"].append(city_pop)
        data["job"].append(job)
        data["dob"].append(dob)
        data["trans_num"].append(trans_num)
        data["unix_time"].append(unix_time)
        data["merch_lat"].append(round(merch_lat, 4))
        data["merch_long"].append(round(merch_long, 4))
        data["is_fraud"].append(is_fraud)
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    generate_sparkov_synthetic_data(60000, os.path.join("dataset", "fraudTrain.csv"))
    generate_sparkov_synthetic_data(15000, os.path.join("dataset", "fraudTest.csv"))
