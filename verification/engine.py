import re
from datetime import datetime

class VerificationEngine:
    def __init__(self, registry):
        self.registry = registry
    
    def verify(self, detected_plate, detected_color, detected_type):
        """Verify detected information against database"""
        flags = []
        status = "verified"
        
        # Query database
        db_info = self.registry.query(detected_plate)
        
        if not db_info:
            flags.append("Plate not found in registry")
            status = "suspicious"
            return {"status": status, "flags": flags, "db_info": None}
        
        # Color mismatch
        if detected_color and db_info['color'].lower() != detected_color.lower():
            flags.append(f"Color mismatch: detected {detected_color}, registered {db_info['color']}")
            status = "suspicious"
        
        # Type mismatch
        if detected_type and db_info['type'].lower() != detected_type.lower():
            flags.append(f"Type mismatch: detected {detected_type}, registered {db_info['type']}")
            status = "suspicious"
        
        # Plate format validation
        if not self.validate_plate_format(detected_plate):
            flags.append("Invalid plate format")
            status = "suspicious"
        
        # Check validity
        if 'plate_details' in db_info:
            validity = datetime.strptime(db_info['plate_details']['validity'], '%Y-%m-%d')
            if validity < datetime.now():
                flags.append("Registration expired")
                status = "suspicious"
        
        return {
            "status": status,
            "flags": flags if flags else ["All checks passed"],
            "db_info": db_info
        }
    
    def validate_plate_format(self, plate):
        """Validate Bangladeshi plate format"""
        # Simple validation - can be enhanced
        return len(plate) > 5
    
    def check_travel_time(self, plate, current_location, current_time):
        """Check if travel time between locations is realistic"""
        history = self.registry.get_movement_history(plate, limit=1)
        
        if not history:
            return True
        
        last_location = history[0]
        time_diff = (current_time - datetime.fromisoformat(last_location['timestamp'])).total_seconds() / 60
        
        # Simplified: assume minimum 10 minutes between different locations
        if last_location['location'] != current_location and time_diff < 10:
            return False
        
        return True
