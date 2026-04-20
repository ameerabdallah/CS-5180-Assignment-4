#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
import re

import numpy as np
# importing required libraries
import pandas as pd
from typing import cast

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# Helper function: tokenize text and remove stopwords only
# ---------------------------------------------------------
def preprocess(input_text: str) -> list[str]:
    return [t for t in re.findall(r'\b\w+\b', input_text.lower()) if t not in ENGLISH_STOP_WORDS]

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# Files:
#   docs.csv
#   queries.csv
#   relevance_judgments.csv
# --> add your Python code here

# noinspection PyArgumentList
docs_dict: dict[str, str] = cast(pd.DataFrame, pd.read_csv('docs.csv')).set_index('doc_id')['text'].to_dict()
# noinspection PyArgumentList
queries_dict: dict[str, str] = cast(pd.DataFrame, pd.read_csv('queries.csv')).set_index('query_id')['query_text'].to_dict()

# set dataframe judgment column to bool (true if R false otherwise)
df = pd.read_csv('relevance_judgments.csv', dtype=str)
judgments: dict[str, dict[str, bool]] = {}
for query_id, doc_id, judgment in df.itertuples(index=False, name=None):
    judgments.setdefault(query_id, {})[doc_id] = (judgment == 'R')

# ---------------------------------------------------------
# 2. Build the BM25 index for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# Steps:
#   1. preprocess each document
#   2. store tokenized documents in a list
#   3. create the BM25 model
# --> add your Python code here

doc_ids = []
docs_list: list[list[str]] = []
for doc_id, text in docs_dict.items():
    doc_ids.append(doc_id)
    docs_list.append(preprocess(text))

bm25 = BM25Okapi(corpus=docs_list)

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# Suggested structure:
#   - for each query:
#       1. preprocess the query
#       2. compute BM25 scores for all documents
#       3. rank documents by score in descending order
#       4. retrieve the relevant documents for that query
#       5. compute AP
# --> add your Python code here

ap_values: dict[str, float] = {}

for query_id, query_text in queries_dict.items():
    query_tokens = preprocess(query_text)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1]
    ranked_doc_ids = [doc_ids[i] for i in ranked_indices]

    relevant = judgments.get(query_id, {})
    relevant_docs = {d for d, is_relevant in relevant.items() if is_relevant}
    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    # Suggested steps:
    #   - initialize variables
    #   - go through the ranked documents
    #   - whenever a relevant document is found:
    #         precision = (# relevant found so far) / (current rank position)
    #         add precision to the running sum
    #   - AP = sum of precisions / total number of relevant documents
    #   - if there are no relevant documents, AP = 0
    # store the AP value for this query (use any data structure you prefer)
    if not relevant_docs:
        ap_values[query_id] = 0.0
        continue

    num_hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_sum += num_hits / rank

    ap_values[query_id] = precision_sum / len(relevant_docs)


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here
sorted_queries = sorted(ap_values.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here
for query_id, ap in sorted_queries:
    print(f"{query_id}: {ap:.4f}")