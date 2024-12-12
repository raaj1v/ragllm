########working csvfile###############
# import pandas as pd
# import os
# import shutil
# import logging
# from datetime import datetime

# def setup_logging():
#     logging.basicConfig(filename='delete_folders.log', level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')

# def delete_folders(csv_path, folder_column, root_folder):
#     setup_logging()
#     logging.info("Script started")
    
#     # Read the CSV file
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         logging.error(f"Error reading CSV file: {e}")
#         return
    
#     # Extract folder numbers from the specified column
#     try:
#         folder_numbers = df[folder_column].astype(str).tolist()
#     except KeyError as e:
#         logging.error(f"Column '{folder_column}' not found in CSV: {e}")
#         return
    
#     # Iterate through each folder in the root directory
#     for folder_name in os.listdir(root_folder):
#         folder_path = os.path.join(root_folder, folder_name)
        
#         # Check if folder_name matches any folder number in the list
#         if folder_name in folder_numbers and os.path.isdir(folder_path):
#             try:
#                 # Delete the folder
#                 shutil.rmtree(folder_path)
#                 logging.info(f"Deleted folder: {folder_path}")
#             except Exception as e:
#                 logging.error(f"Error deleting folder {folder_path}: {e}")
    
#     logging.info("Script finished")

# # Usage
# csv_path = f'/data/QAAPI/tenderprocid.csv'
# folder_column = 'tenderprocid'
# root_folder = f'/data/QAAPI/txtfolder'

# if __name__ == "__main__":
#     delete_folders(csv_path, folder_column, root_folder)
    
    
# # 0 10 * * * /usr/bin/python3 //data/QAAPI/deletetxtfolder.py

#********************************************************************#




#####json working#############
# import os
# import json
# import shutil
# import logging
# def setup_logging():
#     logging.basicConfig(filename='delete_folders.log', level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
# def delete_matching_folders(folder_path, json_file):
#     # Setup logging
#     setup_logging()
#     logging.info("Script started")
#     # Load JSON file
#     with open(json_file, 'r') as f:
#         matching_folders_ids = json.load(f)
#     # Convert folder IDs to strings
#     matching_folders_ids = [str(folder_id) for folder_id in matching_folders_ids]
#     # Iterate over each folder in the directory
#     for root, dirs, files in os.walk(folder_path):
#         for folder in dirs:
#             # Check if the folder name matches any entry in the JSON
#             if folder in matching_folders_ids:
#                 folder_path_to_delete = os.path.join(root, folder)
#                 # Delete the folder and its contents
#                 print("Deleting folder:", folder_path_to_delete)
#                 try:
#                     shutil.rmtree(folder_path_to_delete)
#                     logging.info(f"Deleted folder: {folder_path_to_delete}")
#                 except OSError as e:
#                     print("Error deleting folder:", e)
#                     logging.error(f"Error deleting folder {folder_path_to_delete}: {e}")
#                     continue  # Skip to the next folder
# # Example usage
# folder_path = '/data/QAAPI/txtfolder'
# json_file = '/data/QAAPI/deleted_tenders/2024-06-09.json'
# delete_matching_folders(folder_path, json_file)

#*********************************#


###### working json file with datewise #####
import os
import json
import shutil
import logging
from datetime import datetime
def setup_logging():
    logging.basicConfig(filename='delete_folders.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
def get_latest_json_file(folder_path):
    # Get list of JSON files in the folder
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    # Get the modification time of each file
    file_times = [(file, os.path.getmtime(os.path.join(folder_path, file))) for file in json_files]
    # Sort files by modification time
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)
    # Return the latest JSON file
    if sorted_files:
        return sorted_files[0][0]
    else:
        return None
def delete_matching_folders(folder_path, json_folder):
    # Setup logging
    setup_logging()
    logging.info("Script started")
    # Get the latest JSON file
    json_file = get_latest_json_file(json_folder)
    if json_file is None:
        logging.error("No JSON files found in the folder")
        return
    # Load JSON file
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, 'r') as f:
        matching_folders_ids = json.load(f)
    # Convert folder IDs to strings
    matching_folders_ids = [str(folder_id) for folder_id in matching_folders_ids]
    # Iterate over each folder in the directory
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs:
            # Check if the folder name matches any entry in the JSON
            if folder in matching_folders_ids:
                folder_path_to_delete = os.path.join(root, folder)
                # Delete the folder and its contents
                print("Deleting folder:", folder_path_to_delete)
                try:
                    shutil.rmtree(folder_path_to_delete)
                    logging.info(f"Deleted folder: {folder_path_to_delete}")
                except OSError as e:
                    print("Error deleting folder:", e)
                    logging.error(f"Error deleting folder {folder_path_to_delete}: {e}")
                    continue  # Skip to the next folder
# Example usage
folder_path = "/data/MBBS_pharma/parallel_processing/tenderindex"
json_folder = "/data/QAAPI/deleted_tenders copy_"

delete_matching_folders(folder_path, json_folder)
