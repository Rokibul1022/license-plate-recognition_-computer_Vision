import sqlite3
from datetime import datetime

class VehicleRegistry:
    def __init__(self, db_path="database/vehicle_registry.db"):
        self.db_path = db_path
    
    def query(self, plate_number):
        """Query vehicle information by plate number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get vehicle info
        cursor.execute('''
            SELECT v.plate_number, v.color, v.type, o.name, o.license_number, o.phone, o.city
            FROM vehicles v
            JOIN owners o ON v.owner_id = o.owner_id
            WHERE v.plate_number = ?
        ''', (plate_number,))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return None
        
        vehicle_info = {
            "plate_number": result[0],
            "color": result[1],
            "type": result[2],
            "owner": {
                "name": result[3],
                "license": result[4],
                "phone": result[5],
                "city": result[6]
            }
        }
        
        # Get plate details
        cursor.execute('''
            SELECT district, registration_year, validity
            FROM plates
            WHERE plate_number = ?
        ''', (plate_number,))
        
        plate_info = cursor.fetchone()
        if plate_info:
            vehicle_info["plate_details"] = {
                "district": plate_info[0],
                "registration_year": plate_info[1],
                "validity": plate_info[2]
            }
        
        # Get last location
        cursor.execute('''
            SELECT camera_id, timestamp, location
            FROM locations
            WHERE plate_number = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (plate_number,))
        
        location = cursor.fetchone()
        if location:
            vehicle_info["last_location"] = {
                "camera": location[0],
                "timestamp": location[1],
                "location": location[2]
            }
        
        conn.close()
        return vehicle_info
    
    def get_movement_history(self, plate_number, limit=10):
        """Get movement history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT camera_id, timestamp, location
            FROM locations
            WHERE plate_number = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (plate_number, limit))
        
        history = cursor.fetchall()
        conn.close()
        
        return [{"camera": h[0], "timestamp": h[1], "location": h[2]} for h in history]
