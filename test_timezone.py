from datetime import datetime
import pytz

def get_indian_time():
    """Get current time in Indian Standard Time (IST)"""
    utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.astimezone(ist)

# Test the function
print("UTC Time:", datetime.utcnow())
print("IST Time:", get_indian_time())
print("IST Time (formatted):", get_indian_time().strftime("%Y-%m-%d %H:%M:%S %Z"))
