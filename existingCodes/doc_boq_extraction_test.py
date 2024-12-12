import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')


import os
from docx import Document
import subprocess
# Count occurrences of keywords in a DataFrame
def count_keyword_occurrences(df):
    similarity_threshold = 0.6
    
    keywords = [
        'bill of quantity', 'qty', "inr", 'total price', "weight",'quantity', 'quintal', 
        'quantity in quintal', 'total quantity', 'rate', 'price', 'base rate',
        'offer rate', 'rate per quintal', 'rate in rs.', 'rate per unit', 'item rate', 'rate per item',
        'price per unit', 'schedule of rates', 'total cost', 'cost', 'total base cost',
        'total offer cost', 'price in rs.', 'overall rate', 'aggregate rate', 'extended amount', 'landed cost',
        'final rate', 'total charges', 'lumpsum cost', 'lump-sum', 'item', 'description', 
        'item/activity', 'deliverables', 'item description', 'item name',
        'estimated rate','item type','item/category','drug name','nomenclature','item title'
        'name of works','estimate value','category of works','actions','merged(yes/no)','merged (yes/no)'
        ,'Created Date ','uom'
    ]

    text_data = df.astype(str).values.flatten()
    text_data = [item.lower() for item in text_data if item.strip() != '']

    if not text_data:
        return 0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data + keywords)
    
    cosine_similarities = cosine_similarity(vectors[:len(text_data)], vectors[len(text_data):])
    
    counts = (cosine_similarities.max(axis=0) >= similarity_threshold).sum()
    return counts

def convert_doc_to_docx(file_path):
    output_file = file_path.replace('.doc', '.docx')
    subprocess.run(['unoconv', '-f', 'docx', file_path])
    # Check if the output file exists after conversion
    if os.path.exists(output_file):
        print(f"Successfully converted {file_path} to {output_file}")
        return output_file
    else:
        print(f"Conversion failed: {output_file} not found.")
        return None
    
# Extract tables from DOCX files
def extract_tables_from_docx(file_path):
    from docx import Document
    tables_data = []
    
    doc = Document(file_path)
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables_data.append(pd.DataFrame(table_data))
        
    return tables_data

# Extract tables from DOC files
def extract_tables_from_doc(file_path):
    import win32com.client
    
    tables_data = []
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(file_path)

    for table in doc.Tables:
        table_data = []
        for row in table.Rows:
            row_data = [cell.Range.Text.strip().replace('\r\x07', '') for cell in row.Cells]
            table_data.append(row_data)
        tables_data.append(pd.DataFrame(table_data))

    doc.Close(False)
    word.Quit()
    
    return tables_data

# Function to get the most relevant table
def get_most_relevant_table_doc(file_path):
    extracted_tables = []
    
    # Determine file type
    if file_path.lower().endswith('.docx'):
        tables = extract_tables_from_docx(file_path)
    elif file_path.lower().endswith('.doc'):
        cpn_file=convert_doc_to_docx(file_path)
        tables = extract_tables_from_docx(cpn_file)
    else:
        raise ValueError("Unsupported file type: Only DOC and DOCX files are supported.")

    # Process each extracted table
    for df in tables:
        # Skip empty DataFrames
        if df.empty:
            continue

        # Skip tables with unwanted headers or content
        unwanted_headers = ["annexure name", "bid form", "annexure 01", 'clause no.', 'schedule a', 'date and time', 'pg.ref,']
        if any(header.lower() in df.iloc[0].astype(str).str.lower().values for header in unwanted_headers):
            continue

        keyword_count = count_keyword_occurrences(df)

        if keyword_count > 0:
            extracted_tables.append((keyword_count, df))

    # Sort tables by the number of relevant keywords (descending order)
    extracted_tables.sort(key=lambda x: x[0], reverse=True)

    # Extract the most relevant table
    if extracted_tables:
        most_relevant_table = extracted_tables[0][1]
        # most_relevant_table = extracted_tables[1][1]
        
        if most_relevant_table.empty:
            return None
        
        # Make the second row the header
        if most_relevant_table.shape[0] > 1:
            new_header = most_relevant_table.iloc[0]  # First row as header
            most_relevant_table = most_relevant_table[1:]  # Drop the first row
            most_relevant_table.columns = new_header  # Set the new header
            most_relevant_table.reset_index(drop=True, inplace=True)
            most_relevant_table.dropna(axis=1, how='all', inplace=True)
            most_relevant_table.fillna("-", inplace=True)

        if most_relevant_table.empty or (most_relevant_table.replace("-", "").dropna(how='all').empty):
            return None

        return most_relevant_table
    else:
        return None

