# Importing relevant libraries
# Ensure the following libraries are present numpy, collections, re, sklearn.
import json
from collections import Counter
import re
import numpy as np
from scipy.linalg import svd
from numpy import dot
from numpy.linalg import norm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Importing the dataset, dataset used can be found here - http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz 
# counts is a dictionary of word and it's frequency count
reviews = [json.loads(line) for line in open('reviews.json','r')]
K = 30
MAX_COUNT = 3000
counts = Counter()
splitreviews = []

# Preprocessing dataset, removal of punctuations and making the text lowercase
linenum = 0
for review in reviews:
    text = re.sub("\W"," ",review["reviewText"])
    text = text.lower()
    linesplit = []
    for word in text.split():
        counts[word] += 1
        linesplit.append(word)
    splitreviews.append(linesplit)
    linenum = linenum + 1
    if linenum == MAX_COUNT:
        break
        

vocabulary = counts.keys()

index = 0
for word in vocabulary:
    counts[word] = index
    index = index + 1


matrix = np.zeros((len(vocabulary), len(vocabulary)))

# Building the Co-occurrence Matrix  
for review in splitreviews:
    for i in range(0,len(review)-1):
        matrix[counts[review[i]]][counts[review[i+1]]] += 1
        matrix[counts[review[i+1]]][counts[review[i]]] += 1  
# Applying svd on the Co-occurrence Matrix 
U, D, VT = svd(matrix,full_matrices=False)

word_embeddings = {}
index = 0
for word in vocabulary:
    word_embeddings[word] = U[index][:K]
    index = index + 1

#for word in word_embeddings.keys():
#    print("{"+'"word":{},"word_embeddings":{}'.format(word,word_embeddings)+"}")

# Function to find the top 10 most similar words for a given word
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

top = find_word_embeddings("camera")
print(top)


keys = ['camera', 'product', 'good', 'strong', 'look']

# Code for tsne plots for word embeddings for words similar to ['camera', 'product', 'good', 'strong', 'look']
embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for _ , similar_word in find_word_embeddings(word):
        words.append(similar_word)
        embeddings.append(word_embeddings[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

tsne_model_en_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("model1.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)