import pandas as pd
import csv

nutrition_data = pd.read_csv('nutrition101.csv')

print(nutrition_data['dosa'])