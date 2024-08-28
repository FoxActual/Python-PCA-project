import os
import pandas as pd
from datetime import datetime


# Function to parse the .inf file
def parse_inf(file_path):
    with open(file_path, 'r') as file:
        data = {}
        current_header = None
        for line in file:
            # Check if the line is a header
            if line.strip().endswith(':'):
                current_header = line.strip()[:-1]  # Remove the colon from the header
            elif line.strip() and current_header:
                # Split the line into key and value using tabs
                parts = line.split('\t')
                key = parts[0].strip()
                value = '\t'.join(parts[1:]).strip() if len(parts) > 1 else ""

                # Create a unique column name using the header and key
                column_name = f"{current_header}_{key}"

                # Add the key-value pair to the data dictionary
                data[column_name] = [value]

    return pd.DataFrame(data)


def find_and_extract_extern_inf(parent_folder, existing_df):
    file_data = {'Subfolder': [], 'Extract_Date': []}

    for root, dirs, files in os.walk(parent_folder):
        for dir_name in dirs:
            if dir_name.endswith("_Synapt XS.PRO"):
                print(f"Found matching subfolder: {dir_name}")

                data_folder = os.path.join(root, dir_name, "Data")

                if os.path.exists(data_folder):
                    print(f"Found 'Data' subfolder: {data_folder}")

                    for data_root, data_dirs, data_files in os.walk(data_folder):
                        for subfolder_name in data_dirs:
                            if subfolder_name.endswith(
                                    "_POS FINAL BEAM HR MODE SYNAPT XS 01.raw") or subfolder_name.endswith(
                                    "_NEG FINAL BEAM HR MODE SYNAPT XS 01.raw"):
                                subfolder_path = os.path.join(data_root, subfolder_name)
                                print(f"Found matching subfolder: {subfolder_path}")

                                for subfolder_root, subfolder_dirs, subfolder_files in os.walk(subfolder_path):
                                    for raw_file in subfolder_files:
                                        if raw_file == "_extern.inf":
                                            extern_inf_path = os.path.join(subfolder_root, raw_file)
                                            print(f"Found '_extern.inf' file: {extern_inf_path}")

                                            # Parse the contents of the .inf file and add to the DataFrame
                                            inf_df = parse_inf(extern_inf_path)

                                            # Append to the overall data dictionary
                                            for key, value in inf_df.items():
                                                if key not in file_data:
                                                    file_data[key] = []
                                                file_data[key].extend(value)

                                            # Add the 'Subfolder' value for identification
                                            file_data['Subfolder'].extend([subfolder_name] * len(inf_df))

                                            # Add the 'Extract_Date' value
                                            extract_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            file_data['Extract_Date'].extend([extract_date] * len(inf_df))

    # Create a DataFrame for the new data
    new_data_df = pd.DataFrame(file_data)

    # Filter out rows that are already in the existing DataFrame
    new_data_df = new_data_df[~new_data_df.isin(existing_df.to_dict(orient='list')).all(1)]

    return new_data_df


def update_database_prompt(last_update_date):
    current_date = datetime.now()
    days_since_update = (current_date - last_update_date).days

    if days_since_update >= 90:
        print(f"It has been {days_since_update} days since the database was last updated. Updating database.")
        return True
    else:
        print(f"It has been {days_since_update} days since the database was last updated.")
        response = input("Would you like to update the database now? (Yes/No): ").lower()
        return response == 'yes'


def main():
    parent_folder_path = r'\\tu-server-vfs01\Instrument Data\To Backup\Synapt XS'
    csv_path = 'Synapt_Diag_Database.csv'

    # Load existing database if it exists
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        last_update_date = datetime.strptime(existing_df['Extract_Date'].max(), "%Y-%m-%d %H:%M:%S")
    else:
        existing_df = pd.DataFrame()
        last_update_date = datetime.min

    # Check if the user wants to update the database
    if update_database_prompt(last_update_date):
        # Update the database
        new_data_df = find_and_extract_extern_inf(parent_folder_path, existing_df)

        # Concatenate existing and new data
        final_df = pd.concat([existing_df, new_data_df], ignore_index=True)

        # Save the DataFrame to a CSV file
        final_df.to_csv(csv_path, index=False)
        print("Database updated successfully.")
    else:
        print("Skipping database update.")


if __name__ == "__main__":
    main()
