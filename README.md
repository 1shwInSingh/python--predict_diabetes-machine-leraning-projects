**## To use this program in linux ##**

1. Prepare Your Environment-
   
# Create and enter the project directory
mkdir diabetes_project && cd diabetes_project

# Install the virtual environment package (if you don't have it)
sudo apt update
sudo apt install python3-venv -y

2. Set Up a Virtual Environment

# Create the environment named 'venv'
python3 -m venv venv

# Activate it
source venv/bin/activate

3. Install Required Libraries

pip install pandas numpy scikit-learn matplotlib seaborn

5. Get the Data

# Example: Downloading a common version of the Pima dataset
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv -O diabetes.csv

Then run the file.

**## To use this program in vs-code in windows ##**

1. Create a Virtual Environment (The Windows Way)

Using a virtual environment on Windows prevents "Library Hell" where different projects clash.

(1) Open the Integrated Terminal in VS Code: Press Ctrl + `  (backtick) or go to Terminal > New Terminal.
(2) Type the following commands:

# Create the environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

2. Install Libraries
In that same terminal at the bottom of VS Code, run:

pip install pandas scikit-learn numpy

Then run the file.
