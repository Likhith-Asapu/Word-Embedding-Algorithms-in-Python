# Description 

This repo contains the code for implementation of word embeddings from scratch in python using two methods:

- Frequency-based Embedding - Co-occurrence Matrix method to obtain word embeddings of words occuring in a given corpus.
- Prediction-based Embedding - Word2vec method used for training words representations. Here it is implemented using CBOW method.

# Requirements
- numpy
- collections
- re
- sklearn
- gensim

# Instructions

The models were trained on the following data [LINK](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz)

`python3 part1.py` - To run model1 which uses co-occurrence matrix and svd
`python3 part2.py` - To run model2 which uses Word2ve CBOW model.
Link for embeddings -
[https://drive.google.com/drive/folders/1cK0aUM3likmKcisz2nK9yQlyPqBIioHi?usp=sharing](https://drive.google.com/drive/folders/1cK0aUM3likmKcisz2nK9yQlyPqBIioHi?usp=sharing)

# Code Explanation

## Model 1

#### Step 1 - Construction of co-occurance matrix


```python
for review in splitreviews:
    for i in range(0,len(review)-1):
        matrix[counts[review[i]]][counts[review[i+1]]] += 1
        matrix[counts[review[i+1]]][counts[review[i]]] += 1 
```

Where matrix is a $vocabsize \times vocabsize$ matrix were all entries are intialised to 0. Split reviews contains sentences tokenised.

Example of co-occurance matrix shown below.

1. I enjoy flying.
2. I like NLP.
3. I like deep learning.

The co-occurance matrix for these sentences is $X$ where 

#### Step 2 - Singular Value Decomposition of the co-occurance matrix.

```python
from scipy.linalg import svd
U, D, VT = svd(matrix,full_matrices=False)
```

#### Step 3 - Obtaining the word embeddings from the SVD matrix.

```python
word_embeddings = {}
index = 0
for word in vocabulary:
    word_embeddings[word] = U[index][:K]
    index = index + 1
```

word_embeddings is a dictionary where the keys are the words are values are thier embeddings

To find the top 10 most similar words for a given word use the function find_word_embeddings

```python
def find_word_embeddings(searchword):
    topscore = 0
    topword = " "
    top = []
    for i in range(10):
        top.append([0," "])

    for word in vocabulary:
        a = word_embeddings[searchword]
        b = word_embeddings[word]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        index = 0
        for item in top:
            if cos_sim > item[0] and word != searchword:
                top.insert(index,[cos_sim,word])
                top.pop(10)
                break
            index += 1
    return top
# Example
top = find_word_embeddings("camera")
print(top)
```

# Results

TSNE plots for Model 1(Co-occurance Matrix) for the words 'camera', 'product', 'good', 'strong' and 'look'.

TSNE plots for Model 2(CBOW Word2vec) for the words 'camera', 'product', 'good', 'strong' and 'look'.