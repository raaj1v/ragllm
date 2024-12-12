import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

import requests

def count_keyword_occurrences(df):
    # Define relevant keywords and similarity threshold
    similarity_threshold = 0.6
    
    keywords = [
        'bill of quantity', 'qty', "inr", 'total price', "weight",'quantity', 'quintal', 
        'quantity in quintal', 'total quantity', 'rate', 'price', 'base rate',
        'offer rate', 'rate per quintal', 'rate in rs.', 'rate per unit', 'item rate', 'rate per item',
        'price per unit', 'schedule of rates', 'total cost', 'cost', 'total base cost',
        'total offer cost', 'price in rs.', 'overall rate', 'aggregate rate', 'extended amount', 'landed cost',
        'final rate', 'total charges', 'lumpsum cost', 'lump-sum', 'item', 'description', 
        'item/activity', 'deliverables', 'item description', 'item name',
        'estimated rate','item type','item/category','drug name','nomenclature','item title', 'Name of work'
    ]

    ### chnages,'particulars','activity ---> charges kadhl   
    # nomenclature ======> addd kel
    
    # Flatten DataFrame to a list of strings
    text_data = df.astype(str).values.flatten()
    text_data = [item.lower() for item in text_data if item.strip() != '']

    if not text_data:
        return 0

    # Compute TF-IDF vectors for text data and keywords
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data + keywords)
    
    # Compute cosine similarity between the text data and keywords
    cosine_similarities = cosine_similarity(vectors[:len(text_data)], vectors[len(text_data):])
    
    # Count the number of relevant keywords with a similarity above the threshold
    counts = (cosine_similarities.max(axis=0) >= similarity_threshold).sum()
    return counts

def get_most_relevant_table(pdf_path):
    extracted_tables = []
    
    # Open the PDF file using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        current_table = None
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            # Extract tables from the current page
            tables = page.extract_tables()
            
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)

                    # # Skip if the first row contains unwanted headers
                    # if "Annexure Name" in df.iloc[0].values:
                    #     continue

                    # Skip tables with unwanted headers or content
                    unwanted_headers = ["annexure name", "bid form", "annexure 01",'clause no.',
                'date and time','pg.ref,']  # Add more if needed
                    # if any(header in df.iloc[0].values for header in unwanted_headers):
                    if any(header.lower() in df.iloc[0].astype(str).str.lower().values for header in unwanted_headers):

                        continue
                    
                    # If the current table is None, this is the start of a new table
                    if current_table is None:
                        current_table = df
                    else:
                        # Check if the table should be appended to the current table
                        if df.shape[1] == current_table.shape[1]:
                            current_table = pd.concat([current_table, df], ignore_index=True)
                        else:
                            # Count keyword occurrences and save the table if relevant
                            keyword_count = count_keyword_occurrences(current_table)
                            if keyword_count > 0:
                                extracted_tables.append((keyword_count, current_table))
                            current_table = df
            else:
                # Save the current table when no tables are found on the page
                if current_table is not None:
                    keyword_count = count_keyword_occurrences(current_table)
                    if keyword_count > 0:
                        extracted_tables.append((keyword_count, current_table))
                    current_table = None

        # Append the last table if there is one
        if current_table is not None:
            keyword_count = count_keyword_occurrences(current_table)
            if keyword_count > 0:
                extracted_tables.append((keyword_count, current_table))

    # Sort tables by the number of relevant keywords (descending order)
    extracted_tables.sort(key=lambda x: x[0], reverse=True)

    # Extract the most relevant table from tuples
    if extracted_tables:
        most_relevant_table = extracted_tables[0][1]
        # most_relevant_table.replace(None, "-", inplace=True)

        if most_relevant_table.empty:
            return None
        
        # Make the second row the header
        if most_relevant_table.shape[0] > 1:
            new_header = most_relevant_table.iloc[0]  # Second row as header
            most_relevant_table = most_relevant_table[1:]  # Drop the first two rows
            most_relevant_table.columns = new_header  # Set the new header
            most_relevant_table.reset_index(drop=True, inplace=True)  # Reset index
            most_relevant_table=most_relevant_table.fillna("-")
            most_relevant_table.dropna(axis=1, how='all', inplace=True)  # Drop empty columns
            most_relevant_table.dropna(axis=0, how='all', inplace=True)  # Drop empty rows
            
        # Check if the DataFrame still has valid data (i.e., contains no content)

        if most_relevant_table.empty or (most_relevant_table.replace("-", "").dropna(how='all').empty):
            return None

        return most_relevant_table
    else:
        return None