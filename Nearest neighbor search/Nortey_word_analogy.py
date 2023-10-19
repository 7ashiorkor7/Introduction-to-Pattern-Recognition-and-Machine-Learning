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
        word = vals[0]
        vectors[word] = np.array(vectors[vals[0]])

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

# Function to find the nearest neighbors
def find_nearest_neighbors(vector, exclude_words=[], top_n=2):
    distances = {}
    for word, vec in vectors.items():
        if word not in exclude_words:
            distance = np.linalg.norm(vector - vec)
            distances[word] = distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    return sorted_distances[:top_n]


# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        words = input_term.split("-")
        if len(words) != 3:
            print("Please enter three words.")
            continue

        # Get word vectors for the input words
        try:
            x = vectors[words[0]]
            y = vectors[words[1]]
            z = vectors[words[2]]
        except KeyError:
            print("One or more words not found in vocabulary.")
            continue

        # Calculate the analogy vector
        analogy_vector = z + (y - x)

        # Find nearest neighbors for the analogy vector
        nearest_neighbors = find_nearest_neighbors(analogy_vector, exclude_words=words)

        # Print the results
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for word, distance in nearest_neighbors:
            print("%35s\t\t%f\n" % (word, distance))