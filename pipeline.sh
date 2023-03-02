python3 -m venv .venv
source ./.venv/bin/activate
# .\.venv\Scripts\activate      for Windows
pip install -r requirements.txt
python data_creation.py
python model_preparation.py
python model_preprocessing.py
python model_testing.py