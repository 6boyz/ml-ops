import pickle

from datetime import datetime
from pathlib import Path
from typing import Any


def time_now():
   return datetime.now().strftime("%H:%M:%S")

def print_log(string):
   print(f'@{time_now()} {string}')

def save_data(data: Any, path: str, full_path: str) -> None:
   Path(path).mkdir(parents=True, exist_ok=True)
   with open(full_path, 'wb') as f:
      pickle.dump(data, f)

def load_data(full_path: str) -> Any:
   return pickle.load(open(full_path, "rb"))