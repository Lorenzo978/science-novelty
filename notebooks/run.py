import sys
sys.path.insert(1, '../science_novelty/')

import preprocessing
from tqdm.notebook import tqdm
import csv
from embeddings import load_model, get_embedding
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
from joblib import Parallel, delayed
import collections

# Constants
PATH_OUTPUT = '../data/'
PATH_INPUT = '../data/raw/'
STORAGE = 'csv'
CHUNK_SIZE = 50
TOTAL_PAPERS = None


def preProcess(tag):
    ## Increase the max size of a line reading, otherwise an error is raised
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    print('Get the number of papers to process...')
    with open('../data/raw/papers_raw_'+tag+'.csv', 'r', encoding = 'utf-8') as file:
        line_count = sum(1 for line in file)

    # Subtract 1 for the header if the CSV has a header
    total_papers = line_count - 1

    print('Preparing for writing...')
    words_write = open('../data/processed/papers_words_'+ tag +'.csv','w')
    words_write.write('PaperID,Words_Title,Words_Abstract\n') # write the first line for the headers
    bigrams_write = open('../data/processed/papers_bigrams_'+ tag +'.csv','w')
    bigrams_write.write('PaperID,Bigrams_Title,Bigrams_Abstract\n') # write the first line for the headers
    trigrams_write = open('../data/processed/papers_trigrams_'+ tag +'.csv','w')
    trigrams_write.write('PaperID,Trigrams_Title,Trigrams_Abstract\n') # write the first line for the headers

    print('Processing...')
    with open('../data/raw/papers_raw_'+tag+'.csv', 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter='\t', quotechar='"')
        
        # Skip header
        next(csv_reader)

        for line in tqdm(csv_reader, total = total_papers):
            
            writing_words = line[0] # add the PaperID
            writing_bigrams = line[0] # add the PaperID
            writing_trigrams = line[0] # add the PaperID
            
            ## Assuming that the first two columns are the PaperID and the Date
            for text in [line[2], line[3]]:  # loop over title and abstract
                
                # preprocess text (either title or abstract)            
                unigrams, bigrams, trigrams = preprocessing.process_text(text)
                
                writing_words += ',' + ' '.join(unigrams)
                writing_bigrams += ',' + ' '.join(bigrams)
                writing_trigrams += ',' + ' '.join(trigrams)
                
            words_write.write(writing_words + '\n')
            bigrams_write.write(writing_bigrams + '\n')
            trigrams_write.write(writing_trigrams + '\n')
                
    # close the file
    words_write.close()
    bigrams_write.close()      
    trigrams_write.close()

def textEmbeddings():
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
    # Process the papers

def get_last_processed_index(path_output):
    total_processed = 0
    vectors_path = os.path.join(path_output, 'vectors')
    
    if not os.path.exists(vectors_path):
        return total_processed
    
    for file in os.listdir(vectors_path):
        if file.endswith('.csv') or file.endswith('.npy'):
            file_path = os.path.join(vectors_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                total_processed += sum(1 for line in f) - 1  # Subtract 1 to exclude the header row
    
    return total_processed

def save_vectors(vectors, year, storage, path_output):
    vectors_path = os.path.join(path_output, 'vectors')
    os.makedirs(vectors_path, exist_ok=True)  # Ensure the directory exists
    
    file_path = os.path.join(vectors_path, f'{year}_vectors')
    if storage == 'csv':
        file_path += '.csv'
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, encoding='utf-8', newline='') as writer:
            csv_writer = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if mode == 'w':
                print(f'Creating new file for year {year}...')
                csv_writer.writerow(["PaperID"] + [f"{i}" for i in range(len(vectors[0]) - 1)])  # Adjusted header format
            csv_writer.writerows(vectors)
    elif storage == 'numpy':
        file_path += '.npy'
        vectors = np.array([vec[1:] for vec in vectors])  # Exclude PaperID for numpy storage
        if os.path.exists(file_path):
            existing_vectors = np.load(file_path, allow_pickle=True)
            vectors = np.vstack((existing_vectors, vectors))
        np.save(file_path, vectors)
    else:
        raise ValueError("Unsupported storage format. Use 'csv' or 'numpy'.")

def process_papers(start_index):
    with open(PATH_INPUT + 'papers_raw.csv', 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter='\t', quotechar='"')

        # Skip headers and already processed papers
        print('Already done papers...')
        for _ in tqdm(range(start_index + 2)):
            next(csv_reader)

        for chunk_start in tqdm(range(start_index, TOTAL_PAPERS, CHUNK_SIZE)):
            chunk_data = [line_csv for _, line_csv in zip(range(CHUNK_SIZE), csv_reader)]
            print(chunk_data)
            # Group by year
            papers_by_year = {}
            for data in chunk_data:
                year = int(data[1].split('-')[0])

                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(data)

            # Process each year group
            for year, papers in papers_by_year.items():
                texts = [paper[2] + paper[3] for paper in papers]
                vectors = get_embedding(texts, tokenizer, model)
                print(vectors)
                vectors_with_id = [[paper[0]] + list(vectors[i]) for i, paper in enumerate(papers)]
                save_vectors(vectors_with_id, year, STORAGE, PATH_OUTPUT)


def load_vectors_for_year(year):
    """Load vectors for a specific year using efficient reading."""
    
    file_path = os.path.join(path_vectors, f"{year}_vectors.csv")
    
    if not os.path.exists(file_path):
        return None, None
    
    print(f'Reading {year}...')
    # Load the entire CSV into a single numpy array
    data = xp.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows = 1)
    
    # Check if there is only one paper in the year
    if len(data) == 769:
        papers_ids = [data[0].astype(xp.int64)]
        vectors = [data[1:]]
        
    else:
        # Slice the array to get the desired columns
        papers_ids = data[:, 0].astype(xp.int64)  # Assuming the first column is the PaperId
        vectors = data[:, 1:]  # Assuming the rest of the columns are the vectors

    return papers_ids, vectors

def cosine_similarity(vector_a, vector_b):
    """Simple cosine similarity function"""
    
    norm_a = xp.linalg.norm(vector_a)
    norm_b = xp.linalg.norm(vector_b)
    
    dot_product = xp.dot(vector_a, vector_b)
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

def calculate_similarity_for_chunk(chunk, prior_data):
    """Calculate similarity for a chunk using matrix multiplication."""
    # Normalize the vectors
    chunk_norm = chunk / xp.linalg.norm(chunk, axis=1, keepdims=True)
    prior_data_norm = prior_data / xp.linalg.norm(prior_data, axis=1, keepdims=True)
    
    # Compute cosine similarities using matrix multiplication
    similarities = xp.dot(chunk_norm, prior_data_norm.T)
    
    avg_dists = xp.mean(similarities, axis=1)
    max_dists = xp.max(similarities, axis=1)
    
    return avg_dists, max_dists

def calculate_avg_max_similarity(current_data, prior_data):
    """Calculate average and max cosine similarities for chunks."""
    results = Parallel(n_jobs=N_JOBS)(
        delayed(calculate_similarity_for_chunk)(current_data[i:i+CHUNK_SIZE], prior_data)
        for i in tqdm(range(0, len(current_data), CHUNK_SIZE))
    )
    avg_similarities = xp.concatenate([res[0] for res in results])
    max_similarities = xp.concatenate([res[1] for res in results])
    return avg_similarities, max_similarities

def initialize_output_file():
    """Initialize the output CSV file with headers."""
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['PaperId', 'cosine_max', 'cosine_avg'])

def save_to_csv(papers_ids, avg_similarities, max_similarities):

    """Append results to CSV."""
    with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for paper_id, avg_sim, max_sim in zip(papers_ids, avg_similarities, max_similarities):
            writer.writerow([paper_id, max_sim, avg_sim])


def cosine_distance():
    start_year = 1896
    end_year = 2023
    rolling_data = []
    years = range(start_year, end_year + 1) # +1 to include the last year

    # Initialize the output CSV file
    initialize_output_file()

    for year in tqdm(years):
        papers_ids, current_year_data = load_vectors_for_year(year)
        
        if current_year_data is None:
            continue

        # Add current year data to rolling data
        rolling_data.append((year, current_year_data))
        
        # Remove data that is more than 5 years old
        rolling_data = [(y, data) for y, data in rolling_data if year - y < 6]

        # If there's not enough prior data, skip the calculations for this year
        if len(rolling_data) < 6:
            continue

        # Combine prior years data
        prior_data = xp.vstack([data for y, data in rolling_data if y != year])
        
        print('Calculating similarities for %d...'%(year))
        # Calculate cosine similarities
        avg_year_similarities, max_year_similarities = calculate_avg_max_similarity(current_year_data, prior_data)

        # Save results to CSV
        save_to_csv(papers_ids, avg_year_similarities, max_year_similarities)

def new_word(tag):
    # Count the number of papers
    print('Get the number of papers to process...')
    with open('../data/processed/papers_words_'+ tag +'.csv', 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    total_papers = line_count - 1  # Subtract 1 for the header

    print('Creating the baseline...')
    # Creating a baseline set of words from papers published before the baseline year
    baseline_year = 2000
    baseline = set()

    print('Iterating over the baseline...')
    with open('../data/raw/papers_raw_'+tag+'.csv', 'r', encoding='utf-8') as raw_reader, \
            open('../data/processed/papers_words_'+ tag +'.csv', 'r', encoding='utf-8') as processed_reader:
            
        csv_raw_reader = csv.reader(raw_reader, delimiter='\t', quotechar='"')
        csv_processed_reader = csv.reader(processed_reader, delimiter=',', quotechar='"')

        # Skipping the headers
        next(csv_raw_reader)
        next(csv_processed_reader)
        
        # Iterating over each paper and adding words to the baseline if the paper was published before the baseline year
        for line_raw, line_processed in tqdm(zip(csv_raw_reader, csv_processed_reader), total=total_papers):
            if int(line_raw[1].split('-')[0]) > baseline_year:
                continue
                
            text = set(line_processed[1].split() + line_processed[2].split())
            baseline.update(text)
            
    # Counting the occurrence of new words that are not in the baseline
    counter = collections.Counter()
    paperIds = collections.defaultdict()

    print('Calculating new words...')
    # Reading the processed papers data and counting new words
    with open('../data/processed/papers_words_'+ tag +'.csv', 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
        next(csv_reader)  # Skip header

        for line in tqdm(csv_reader, total=total_papers):
            paperID = int(line[0])
            text = set(line[1].split() + line[2].split())
            
            for token in text:
                if token in baseline:
                    continue
                    
                if token not in counter:
                    counter[token] = 0
                    paperIds[token] = paperID
                else:
                    counter[token] += 1
                    
    print('Exporting the results...')
    # Exporting the results to a new CSV file
    with open('../data/metrics/new_word_'+tag+'.csv', 'w', encoding="utf-8") as writer:
        writer.write('word,PaperID,reuse\n') # Header
        print("---------------------------")
        for token, paperID, reuse in tqdm(zip(counter.keys(), paperIds.values(), counter.values()), total=len(counter)):
            # Filter out if reused only once
            print(paperID)
            if reuse == 0:
                continue
            writer.write(f'{token},{paperID},{reuse}\n')
        print("---------------------------")


def new_bigram(tag):
    # Count the number of papers
    print('Get the number of papers to process...')
    with open('../data/processed/papers_bigrams_'+ tag +'.csv', 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    total_papers = line_count - 1  # Subtract 1 for the header

    print('Creating the baseline...')
    # Creating a baseline set of bigrams from papers published before the baseline year
    baseline_year = 2000
    baseline = set()

    print('Iterating over the baseline...')
    with open('../data/raw/papers_raw_'+tag+'.csv', 'r', encoding='utf-8') as raw_reader, \
            open('../data/processed/papers_bigrams_'+ tag +'.csv', 'r', encoding='utf-8') as processed_reader:
            
        csv_raw_reader = csv.reader(raw_reader, delimiter='\t', quotechar='"')
        csv_processed_reader = csv.reader(processed_reader, delimiter=',', quotechar='"')

        # Skipping the headers
        next(csv_raw_reader)
        next(csv_processed_reader)
        
        # Iterating over each paper and adding bigrams to the baseline if the paper was published before the baseline year
        for line_raw, line_processed in tqdm(zip(csv_raw_reader, csv_processed_reader), total=total_papers):
            if int(line_raw[1].split('-')[0]) > baseline_year:
                continue
                
            text = set(line_processed[1].split() + line_processed[2].split())
            baseline.update(text)
            
    # Counting the occurrence of new bigrams that are not in the baseline
    counter = collections.Counter()
    paperIds = collections.defaultdict()

    print('Calculating new bigrams...')
    # Reading the processed papers data and counting new bigrams
    with open('../data/processed/papers_bigrams_'+ tag +'.csv', 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
        next(csv_reader)  # Skip header

        for line in tqdm(csv_reader, total=total_papers):
            paperID = int(line[0])
            text = set(line[1].split() + line[2].split())
            
            for token in text:
                if token in baseline:
                    continue
                    
                if token not in counter:
                    counter[token] = 0
                    paperIds[token] = paperID
                else:
                    counter[token] += 1
                    
    print('Exporting the results...')
    # Exporting the results to a new CSV file
    with open('../data/metrics/new_bigram_'+tag+'.csv', 'w', encoding="utf-8") as writer:
        writer.write('bigram,PaperID,reuse\n') # Header

        for token, paperID, reuse in tqdm(zip(counter.keys(), paperIds.values(), counter.values()), total=len(counter)):
            # Filter out if reused only once
            if reuse == 0:
                continue

            writer.write(f'{token},{paperID},{reuse}\n')


def new_trigram(tag):
    # Count the number of papers
    print('Get the number of papers to process...')
    with open('../data/processed/papers_trigrams_'+ tag +'.csv', 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    total_papers = line_count - 1  # Subtract 1 for the header

    print('Creating the baseline...')
    # Creating a baseline set of trigrams from papers published before the baseline year
    baseline_year = 2000
    baseline = set()

    print('Iterating over the baseline...')
    with open('../data/raw/papers_raw_'+tag+'.csv', 'r', encoding='utf-8') as raw_reader, \
            open('../data/processed/papers_trigrams_'+ tag +'.csv', 'r', encoding='utf-8') as processed_reader:
            
        csv_raw_reader = csv.reader(raw_reader, delimiter='\t', quotechar='"')
        csv_processed_reader = csv.reader(processed_reader, delimiter=',', quotechar='"')

        # Skipping the headers
        next(csv_raw_reader)
        next(csv_processed_reader)
        
        # Iterating over each paper and adding trigrams to the baseline if the paper was published before the baseline year
        for line_raw, line_processed in tqdm(zip(csv_raw_reader, csv_processed_reader), total=total_papers):
            if int(line_raw[1].split('-')[0]) > baseline_year:
                continue
                
            text = set(line_processed[1].split() + line_processed[2].split())
            baseline.update(text)
            
    # Counting the occurrence of new trigrams that are not in the baseline
    counter = collections.Counter()
    paperIds = collections.defaultdict()

    print('Calculating new trigrams...')
    # Reading the processed papers data and counting new trigrams
    with open('../data/processed/papers_trigrams_'+ tag +'.csv', 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
        next(csv_reader)  # Skip header

        for line in tqdm(csv_reader, total=total_papers):
            paperID = int(line[0])
            text = set(line[1].split() + line[2].split())
            
            for token in text:
                if token in baseline:
                    continue
                    
                if token not in counter:
                    counter[token] = 0
                    paperIds[token] = paperID
                else:
                    counter[token] += 1
                    
    print('Exporting the results...')
    # Exporting the results to a new CSV file
    with open('../data/metrics/new_trigram_'+tag+'.csv', 'w', encoding="utf-8") as writer:
        writer.write('trigram,PaperID,reuse\n') # Header

        for token, paperID, reuse in tqdm(zip(counter.keys(), paperIds.values(), counter.values()), total=len(counter)):
            # Filter out if reused only once
            if reuse == 0:
                continue

            writer.write(f'{token},{paperID},{reuse}\n')


def main(tag):
    preProcess(tag)

    # TEXT EMBEDDINGS
    textEmbeddings()

    # Check if paths exist
    if not os.path.exists(PATH_OUTPUT) or not os.path.exists(PATH_INPUT):
        raise Exception("Input or output path does not exist.")

    # Load the embedding model
    print('Loading the embedding model...')
    tokenizer, model = load_model()

    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using {device.upper()}.")

    # Count the number of papers
    print('Get the number of papers to process...')
    with open(PATH_INPUT + 'papers_raw_'+ tag +'.csv', 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    TOTAL_PAPERS = line_count - 1  # Subtract 1 for the header
    # Get the last processed paper index

    last_processed_index = get_last_processed_index(PATH_OUTPUT)
    print(f"Resuming from paper {last_processed_index + 1}.")
    process_papers(last_processed_index)

    new_word(tag)
    new_bigram(tag)
    new_trigram(tag)


if __name__ == "__main__":
    print("Executing processing for text without summary")
    main("nosummary")
    print("Executing processing for text with summary")
    main("summary")

