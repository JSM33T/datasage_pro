from datetime import datetime
import pytz

def get_indian_time():
    """Get current time in Indian Standard Time (IST) - timezone naive for MongoDB"""
    utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    ist_time = utc_now.astimezone(ist)
    # Return timezone-naive datetime in IST for MongoDB compatibility
    return ist_time.replace(tzinfo=None)

# Test the function
utc_time = datetime.utcnow()
ist_time = get_indian_time()

print("UTC Time:", utc_time)
print("IST Time (timezone-naive):", ist_time)
print("Time difference:", ist_time - utc_time)
print("IST Time (formatted):", ist_time.strftime("%Y-%m-%d %H:%M:%S"))
