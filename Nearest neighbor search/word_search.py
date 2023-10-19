""" Using machine learning techniques, words of any language can be embedded into a high dimensional
Euclidean space where semantically similar words are close to each other.
You are provided a file that contains 400,000 English terms and their 50-dimensional embedding vectors.
You are also provided a code that loads the words and their vectors.
Search similar words 
For any input word, return the three (3) most similar words (the most similar should be the input word
itself). Give results at least to:
• king
• europe
• frog
 """

import random
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for finding similar words
while True:
    input_term = input("\nEnter a word (EXIT to break): ").strip()
    if input_term == 'EXIT':
        break
    elif input_term in vocab:
        input_idx = vocab[input_term]

        # Compute Euclidean distances
        distances = np.sqrt(np.sum((W - W[input_idx])**2, axis=1))

        # Sort distances and get the indices of the smallest distances
        most_similar_indices = np.argsort(distances)

        # Print the most similar words (excluding the input word itself)
        print("\nMost similar words to '{}':".format(input_term))
        for idx in most_similar_indices[:4]:  # Including the input word itself
            if idx != input_idx:
                print("{:<15} (Distance: {:.4f})".format(ivocab[idx], distances[idx]))
    else:
        print("Word not found in vocabulary.")    
