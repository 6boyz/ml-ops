DATE_BIN=$(command -v date)
DATE=`${DATE_BIN} +@%H:%M:%S`

print_log () {
    echo $DATE $*
}

print_log Data download...
mkdir data_default
cd data_default
curl -LJO https://raw.githubusercontent.com/6boyz/ml-ops/master/static-files/house-pricing.csv
cd -

print_log Install Python...
apt install python3

print_log Install pip...
apt install python3-pip

print_log Install virtualenv for Python...
python3 -m pip install --user virtualenv

print_log Create Python venv with name .venv...
python3 -m venv .venv

print_log Activate venv...
source .venv/Scripts/activate

print_log Current version: `python --version`

print_log Install requirements...
pip install -r requirements.txt

print_log Data creation script run...
python data_creation.py

print_log Model preprocessing script run...
python model_preprocessing.py

print_log Model preparation script run...
python model_preparation.py

print_log Model testing script run...
python model_testing.py

print_log Done
read