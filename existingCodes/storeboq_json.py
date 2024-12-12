from datetime import datetime,timedelta
import os
import pandas as pd
from fuzzywuzzy import fuzz
import psycopg2
import json
import nltk
import os
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf_boq_extraction2 import *
# nltk.download('punkt')
import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor  # Correct import


from concurrent.futures import ProcessPoolExecutor  # Import the ProcessPoolExecutor

def should_drop(value):
    if isinstance(value, str):
        return value.startswith("number #")
    return False

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def normalize_text(text):
    if isinstance(text, str):
        return ' '.join(text.lower().split())
    return text


def update_isboq_flag(tc_no, flag_value):
    try:
        connection = psycopg2.connect(
            user="boq",
            password="boq",
            host="localhost",
            port="5432",
            database="tbl_docboq"
        )
        cursor = connection.cursor()
        update_query = """
        UPDATE tbl_document_boq
        SET isboq = %s
        WHERE tc_no = %s;
        """
        cursor.execute(update_query, (flag_value, tc_no))
        connection.commit()
        print(f"'isboq' flag updated to {flag_value} for tc_no: {tc_no}")
    except Exception as error:
        print(f"Error while updating 'isboq' flag: {error}")
    finally:
        if connection:
            cursor.close()
            connection.close()
            
def make_columns_unique(df):
    # while saving the unique column we atre facing the issue for handling this we uses this
    counts = {}
    new_columns = []
    for col in df.columns:
        if col in counts:
            counts[col] += 1
            new_col = f"{col}_{counts[col]}"  # Append a count to the column name
            new_columns.append(new_col)
        else:
            counts[col] = 0
            new_columns.append(col)
    df.columns = new_columns  # Assign the new column names
    return df

import psycopg2
import json

def merge_json(existing_json, new_json):
    try:
        # Log inputs for debugging
        # print(f"Existing JSON: {existing_json}")
        # print(f"New JSON: {new_json}")

        # Handle cases where input is not a valid JSON string
        if isinstance(existing_json, str):
            existing_data = json.loads(existing_json)  # Parse JSON string to Python object
        elif isinstance(existing_json, list):
            existing_data = existing_json  # Already a Python list
        elif existing_json is None:
            existing_data = []  # Default to an empty list
        else:
            raise ValueError(f"Unexpected type for existing_json: {type(existing_json)}")

        if isinstance(new_json, str):
            new_data = json.loads(new_json)  # Parse JSON string to Python object
        elif isinstance(new_json, list):
            new_data = new_json  # Already a Python list
        elif new_json is None:
            new_data = []  # Default to an empty list
        else:
            raise ValueError(f"Unexpected type for new_json: {type(new_json)}")

        # Merge the two lists
        merged_data = existing_data + new_data
        return merged_data  # Return as Python list, not JSON string

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return []  # Default to an empty list
    except Exception as e:
        print(f"Error in merge_json: {e}")
        return []  # Default to an empty list


def save_all_dfs(dfs, tc_no):
    try:
        connection = psycopg2.connect(
            user="boq",
            password="boq",
            host="localhost",
            port="5432",
            database="tbl_docboq"
        )
        cursor = connection.cursor()
        
        if not dfs:
            # No DataFrames were processed, insert tc_no with isboq set to 0
            insert_query = """
            INSERT INTO tbl_document_boq (tc_no, boq_no, extracted_boq, isboq, is_regenerate)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tc_no, boq_no)
            DO NOTHING;
            """
            cursor.execute(insert_query, (tc_no, 0, None, 0, 0))
            connection.commit()
            print(f"No data extracted. 'isboq' flag set to 0 for tc_no: {tc_no}")
            return
        
        for idx, df in enumerate(dfs):
            if df.columns.duplicated().any():
                print(f"Duplicate columns found in DataFrame for tc_no: {tc_no}")
                df = make_columns_unique(df)
                
            new_boq_json = df.to_json(orient='records')
            boq_no = idx + 1
            select_query = "SELECT extracted_boq FROM tbl_document_boq WHERE tc_no = %s AND boq_no = %s;"
            cursor.execute(select_query, (tc_no, boq_no))
            result = cursor.fetchone()

            if result:
                existing_boq_json = result[0]
                if not isinstance(existing_boq_json, (str, list)):  # Ensure valid type
                    print(f"Invalid data type for existing_boq_json: {type(existing_boq_json)}")
                    existing_boq_json = json.dumps([])  # Default to empty JSON string
                merged_boq_json = merge_json(existing_boq_json, new_boq_json)
                # print("merged_boq_json:::", merged_boq_json)
            else:
                merged_boq_json = new_boq_json

            # Insert merged JSON as a proper JSON object (not as a string)
            insert_query = """
            INSERT INTO tbl_document_boq (tc_no, boq_no, extracted_boq)
            VALUES (%s, %s, %s)
            ON CONFLICT (tc_no, boq_no)
            DO UPDATE SET extracted_boq = EXCLUDED.extracted_boq;
            """
            # Ensure that merged_boq_json is a Python list, which will be inserted as a JSON object
            cursor.execute(insert_query, (tc_no, boq_no, json.dumps(merged_boq_json)))

        connection.commit()
        print(f"Data successfully inserted/updated for tc_no: {tc_no}")

        if dfs:
            update_isboq_flag(tc_no, 9)

    except Exception as error:
        print(f"Error while connecting to PostgreSQL: {error}")
    finally:
        if connection:
            cursor.close()
            connection.close()




def find_header_row_1608(df, header_keywords, min_matches=2):
    for i, row in df.iterrows():
        row_str = ' '.join(row.dropna().astype(str).tolist()).lower()
        # Check individual keyword matches
        keyword_matches = [any(fuzz.partial_ratio(keyword, word) > 40 for word in row_str.split()) for keyword in header_keywords]
        if sum(keyword_matches) >= min_matches:  # Heuristic: At least 2-3 keywords should match
            return i
    return None

def name_check_1609(folder_path):
    file_names = os.listdir(folder_path)
    valid_extensions = ('.xlsx', '.csv', '.pdf', '.xls')
    matched_files = [file for file in file_names if file.lower().endswith(valid_extensions)]
    print("Matched files:", matched_files)    
    all_dfs = []  # List to store DataFrames
    # print("list:::::",all_dfs)
    for file_name in matched_files:
        try:
            file_path = os.path.join(folder_path, file_name)
            df = None
            if file_name.lower().endswith('.xlsx') or file_name.lower().endswith('.xls'):
                if file_name.lower().endswith('.xls'):
                    df = pd.read_excel(file_path, engine='xlrd')
                else:
                    df = pd.read_excel(file_path)
                print("XLSX:", file_name)                
            elif file_name.lower().endswith('.csv'):
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                print("CSV:", file_name)                
            elif file_name.lower().endswith('.pdf'):
                df = get_most_relevant_table(file_path)
                if df is not None:
                    print(f"Processed PDF: {file_name}")                        
                    if len(all_dfs) > 1:  # If all_dfs is empty
                        if 'RA Details' in df.columns:
                            print(f"Skipping file with 'RA Details' column: {file_name}")
                            continue  # Skip appending this DataFrame
                    else:                         
                        all_dfs.append(df)
                else:
                    print(f"Failed to extract table from PDF: {file_name}")
                    continue
            else:
                print("Other file type:", file_name)
                continue

            if df is not None and not file_name.lower().endswith('.pdf'):
                header_keywords = [
                    'bill of quantity', 'qty', 'quantity', "inr", "item/activity", 'total price', "item description", "weight",
                    'quantity', 'qty', 'quintal', 'quantity in quintal', 'total quantity', 'rate', 'price', 'base rate',
                    'offer rate', 'rate per quintal', 'rate in rs.', 'rate per unit', 'item rate', 'rate per item',
                    'price per unit', 'price', 'schedule of rates', 'charges', 'total cost', 'cost', 'total base cost',
                    'total offer cost', 'price in rs.', 'overall rate', 'aggregate rate', 'extended amount', 'landed cost',
                    'final rate', 'total charges', 'lumpsum cost', 'lump-sum', 'item', 'description', 'particulars',
                    'activity', 'item/activity', 'deliverables', 'item description', 'item name', 'item title', 'item description'
                ]
                header_row_idx = find_header_row_1608(df, header_keywords)
                if header_row_idx is not None:
                    if header_row_idx != 0:
                        df.columns = df.iloc[header_row_idx]
                        df = df[header_row_idx + 1:].reset_index(drop=True)
                    df = df.apply(lambda x: x.map(normalize_text) if x.dtype == 'object' else x)
                    specific_cols = [0, 1, 3, 4, 5, 12, 52, 53, 54]
                    max_index = df.shape[1] - 1
                    valid_cols = [col for col in specific_cols if col <= max_index]
                    if valid_cols:
                        df = df.iloc[:, valid_cols]
                        rows_to_drop = df.apply(lambda x: x.map(should_drop).any(), axis=1)
                        first_row_to_drop = rows_to_drop.idxmax() if rows_to_drop.any() else None
                        if first_row_to_drop is not None:
                            df = df.drop(index=df.index[:first_row_to_drop + 1])
                            print(f"Dropped rows up to {first_row_to_drop} in file: {file_name}")
                        else:
                            df = df.copy()
                        df.columns = [str(col) for col in df.columns] 
                        df.columns = ['-' if 'Unnamed' in col else col for col in df.columns]
                        df.dropna(axis=1, how='all', inplace=True)
                        df.dropna(axis=0, how='all', inplace=True)
                        all_dfs.append(df)
                    else:
                        print(f"No valid columns found in file: {file_name}")
                else:
                    print(f"Header row not found in file: {file_name}")
        except pd.errors.ParserError:
            print(f"Error reading file: {file_name}")
        except Exception as e:
            print(f"Error reading file: {file_name} - {e}")

    save_all_dfs(all_dfs, os.path.basename(folder_path))
    # print("all:::",all_dfs)
    all_dfs.clear()  # Release memory after processing each folder

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
            
if __name__ == "__main__":
    previous_day = (datetime.now() - timedelta(days=4)).strftime("%d-%m-%y")
    base_path = f"/data/unzipdocument/dailydocument_{previous_day}"
    folder_paths = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    num_workers = 40  # Adjust this number based on your CPU cores
    folder_chunk_size = 80
    for folder_chunk in chunks(folder_paths, folder_chunk_size):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(name_check_1609, folder_chunk)

