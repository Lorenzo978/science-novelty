import os
import pandas as pd
from summarizer import Summarizer
import re
import torch


# Set the maximum number of CPU cores for loky
os.environ['LOKY_MAX_CPU_COUNT'] = '2'

def read_csv(file_path):
    return pd.read_csv(file_path, sep='\t')

def write_csv(file_path, data, header=True, mode='w'):
    data.to_csv(file_path, sep='\t', index=False, header=header, mode=mode)

def generate_summary(text, max_summary_length=500):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Summarizer()
   # model = model.to(device)
    
    summary = model(text, ratio=0.2).strip()  # Apply trimming to the generated summary
    summary = re.sub(' +', ' ', summary)
    summary = summary.replace('\n', ' ')
    summary = summary[:max_summary_length]
    return summary

def process_papers(input_csv, papers_folder, output_csv_without_summary, output_csv_with_summary):
    data = read_csv(input_csv)

    for index in range(len(data)):
        row = data.iloc[index]
        paper_id = row['PaperID']
        txt_file_path = os.path.join(papers_folder, f'{paper_id}.txt')

        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read().strip()  # Apply trimming to the text file content
                # Remove newlines from content
                content = content.replace('\n', ' ')

            # Add print statement to display the current row number
            print(f"Processing row {index + 1}/{len(data)}")

            # Append data to 'paper_without_summary.csv'
            row_without_summary = {'PaperID': paper_id, 'Date': row['Date'], 'Title': row['Title'], 'Abstract': content}
            if os.path.exists(output_csv_without_summary):
                write_csv(output_csv_without_summary, data=pd.DataFrame([row_without_summary]), header=not os.path.exists(output_csv_without_summary), mode='a')
            else:
                write_csv(output_csv_without_summary, data=pd.DataFrame([row_without_summary]), header=not os.path.exists(output_csv_without_summary), mode='w')

            # Generate summary
            summary = generate_summary(content)
            
            # Append data to 'paper_with_summary.csv'
            row_with_summary = {'PaperID': paper_id, 'Date': row['Date'], 'Title': row['Title'], 'Abstract': summary}
            if os.path.exists(output_csv_with_summary):
                write_csv(output_csv_with_summary, data=pd.DataFrame([row_with_summary]), header=not os.path.exists(output_csv_with_summary), mode='a')
            else:
                write_csv(output_csv_with_summary, data=pd.DataFrame([row_with_summary]), header=not os.path.exists(output_csv_with_summary), mode='w')

# Example usage
process_papers(
    input_csv='data/raw/papers_raw.csv',
    papers_folder='papers',
    output_csv_without_summary='papers_raw_nosummary.csv',
    output_csv_with_summary='papers_raw_summary.csv'
)
