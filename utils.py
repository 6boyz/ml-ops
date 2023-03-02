from datetime import datetime

def time_now():
   return datetime.now().strftime("%H:%M:%S")