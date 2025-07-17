from datetime import datetime, timedelta
import pytz

def get_indian_time():
    """Get current time in Indian Standard Time (IST) - timezone naive for MongoDB"""
    # Use timezone-aware UTC time for better accuracy
    utc_now = datetime.now(pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    ist_time = utc_now.astimezone(ist)
    # Return timezone-naive datetime in IST for MongoDB compatibility
    return ist_time.replace(tzinfo=None)

def compensate_mongo_time(mongo_datetime):
    """Compensate for MongoDB time delay by adding 5:30 hours"""
    if mongo_datetime is None:
        return None
    
    # Add 5 hours and 30 minutes compensation
    compensated_time = mongo_datetime + timedelta(hours=5, minutes=30)
    return compensated_time

def format_compensated_time(mongo_datetime):
    """Format MongoDB datetime with compensation for display"""
    if mongo_datetime is None:
        return None
    
    compensated = compensate_mongo_time(mongo_datetime)
    return compensated.strftime("%d/%m/%Y %H:%M:%S")

def test_time_compensation():
    """Test time compensation functionality"""
    print("=== Time Compensation Test ===")
    
    # Simulate a MongoDB datetime that's behind by 5:30 hours
    current_time = get_indian_time()
    print(f"Current IST time: {current_time}")
    
    # Simulate delayed MongoDB time (subtract 5:30 hours)
    delayed_mongo_time = current_time - timedelta(hours=5, minutes=30)
    print(f"Simulated MongoDB time (delayed): {delayed_mongo_time}")
    
    # Apply compensation
    compensated_time = compensate_mongo_time(delayed_mongo_time)
    print(f"Compensated time: {compensated_time}")
    
    # Format for display
    formatted_time = format_compensated_time(delayed_mongo_time)
    print(f"Formatted compensated time: {formatted_time}")
    
    # Check if compensation worked
    time_diff = abs((current_time - compensated_time).total_seconds())
    print(f"Time difference after compensation: {time_diff} seconds")
    
    if time_diff < 60:  # Less than 1 minute difference
        print("✅ Time compensation is working correctly!")
    else:
        print("❌ Time compensation needs adjustment")
    
    return {
        'current_time': current_time,
        'delayed_time': delayed_mongo_time,
        'compensated_time': compensated_time,
        'formatted_time': formatted_time
    }

def get_server_time_info():
    """Get comprehensive server time information"""
    # Current UTC time
    utc_now = datetime.now(pytz.UTC)
    
    # IST time
    ist = pytz.timezone('Asia/Kolkata')
    ist_time = utc_now.astimezone(ist)
    
    # System local time
    local_time = datetime.now()
    
    return {
        'utc_time': utc_now,
        'ist_time': ist_time,
        'ist_time_naive': ist_time.replace(tzinfo=None),
        'local_time': local_time,
        'ist_formatted': ist_time.strftime("%d-%m-%Y %H:%M:%S %Z"),
        'utc_formatted': utc_now.strftime("%d-%m-%Y %H:%M:%S %Z")
    }

if __name__ == "__main__":
    print("=== Server Time Information ===")
    time_info = get_server_time_info()
    for key, value in time_info.items():
        print(f"{key}: {value}")

    print("\n=== Legacy Function Test ===")
    ist_time = get_indian_time()
    print("IST Time (timezone-naive):", ist_time)
    print("IST Time (formatted):", ist_time.strftime("%d-%m-%Y %H:%M:%S"))

    print("\n=== Time Comparisons ===")
    print("UTC vs IST difference:", time_info['ist_time_naive'] - time_info['utc_time'].replace(tzinfo=None))
    print("Expected IST offset: +5:30 hours from UTC")
    
    print("\n")
    test_time_compensation()
