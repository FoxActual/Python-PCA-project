import pandas as pd
import openpyxl
from pathlib import Path

# Replace 'input.csv' and 'output.xlsx' with the actual paths for your input CSV file and desired output Excel file
csv_file_path = 'Synapt_Diag_Database.csv'
excel_file_path = 'Failing_Data.xlsx'


# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)

# Check if the Excel file already exists
if not Path(excel_file_path).is_file():
    # If the file doesn't exist, create a new Excel file with a dummy sheet
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        writer.book = openpyxl.Workbook()
        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        df.to_excel(writer, sheet_name='Sheet1', index=False)
else:
    # If the file exists, overwrite the existing Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        writer.book = openpyxl.Workbook()
        writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        df.to_excel(writer, sheet_name='Sheet1', index=False)

print(f"Excel file created or updated successfully: {excel_file_path}")
