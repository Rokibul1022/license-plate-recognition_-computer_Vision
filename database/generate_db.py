import sqlite3
import random
from datetime import datetime, timedelta

# Bangladeshi names and districts
FIRST_NAMES = ["Rahim", "Karim", "Fatema", "Ayesha", "Jamal", "Nasir", "Sultana", "Begum", "Hasan", "Hussain"]
LAST_NAMES = ["Ahmed", "Rahman", "Khan", "Ali", "Islam", "Hossain", "Chowdhury", "Mia", "Uddin", "Akter"]
DISTRICTS = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet", "Barisal", "Rangpur", "Mymensingh"]
COLORS = ["White", "Black", "Silver", "Red", "Blue", "Green", "Gray"]
VEHICLE_TYPES = ["Sedan", "SUV", "Microbus", "Pickup", "Motorcycle", "CNG", "Bus", "Truck"]

def generate_plate_number(district):
    """Generate realistic BD plate number"""
    metro_codes = {"Dhaka": "মেট্রো", "Chattogram": "মেট্রো"}
    
    if district in metro_codes:
        prefix = f"{district} {metro_codes[district]}"
    else:
        prefix = district
    
    class_letter = random.choice(['গ', 'ক', 'খ', 'ঘ', 'চ'])
    numbers = f"{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    
    return f"{prefix}-{class_letter} {numbers}"

def create_database(db_path="database/vehicle_registry.db", num_records=1000):
    """Create synthetic vehicle registry database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS owners (
            owner_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            license_number TEXT NOT NULL,
            phone TEXT NOT NULL,
            city TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            color TEXT NOT NULL,
            type TEXT NOT NULL,
            owner_id INTEGER,
            FOREIGN KEY (owner_id) REFERENCES owners(owner_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            plate_number TEXT PRIMARY KEY,
            district TEXT NOT NULL,
            registration_year INTEGER NOT NULL,
            validity TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            location TEXT NOT NULL
        )
    ''')
    
    # Generate synthetic data
    print(f"Generating {num_records} synthetic records...")
    
    for i in range(num_records):
        # Owner
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        license_num = f"DL-{random.choice(['DA', 'CH', 'RA'])}-{random.randint(100000, 999999)}"
        phone = f"01{random.randint(700000000, 999999999)}"
        city = random.choice(DISTRICTS)
        
        cursor.execute('''
            INSERT INTO owners (name, license_number, phone, city)
            VALUES (?, ?, ?, ?)
        ''', (name, license_num, phone, city))
        
        owner_id = cursor.lastrowid
        
        # Vehicle
        district = random.choice(DISTRICTS)
        plate_number = generate_plate_number(district)
        color = random.choice(COLORS)
        vehicle_type = random.choice(VEHICLE_TYPES)
        
        try:
            cursor.execute('''
                INSERT INTO vehicles (plate_number, color, type, owner_id)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, color, vehicle_type, owner_id))
            
            # Plate info
            reg_year = random.randint(2015, 2023)
            validity = f"{reg_year + 15}-12-31"
            
            cursor.execute('''
                INSERT INTO plates (plate_number, district, registration_year, validity)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, district, reg_year, validity))
            
            # Location history
            cameras = ["Farmgate-01", "Mohakhali-02", "Gulshan-03", "Dhanmondi-04", "Mirpur-05"]
            for _ in range(random.randint(1, 5)):
                timestamp = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
                camera = random.choice(cameras)
                location = camera.split('-')[0]
                
                cursor.execute('''
                    INSERT INTO locations (plate_number, camera_id, timestamp, location)
                    VALUES (?, ?, ?, ?)
                ''', (plate_number, camera, timestamp, location))
        
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    print(f"Database created at {db_path}")
    print("⚠️ DISCLAIMER: All personal data are synthetically generated for academic research.")

if __name__ == "__main__":
    import os
    os.makedirs("database", exist_ok=True)
    create_database()
