from datetime import datetime

def time_now():
   return datetime.now().strftime("%H:%M:%S")

def print_log(string):
   print(f'@{time_now()} {string}')