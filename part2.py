# Importing relevant libraries
# Ensure the following libraries are present numpy, collections, re, sklearn, gensim.
import numpy as np
import json
from collections import Counter
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm
import math
import gensim.downloader as api
from gensim.models import Word2Vec

# Importing the dataset, dataset used can be found here - http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz 
# counts is a dictionary of word and it's frequency count
reviews = [json.loads(line) for line in open('reviews.json','r')]
K = 30
MAX_COUNT = 1000
CONTEXT_SIZE = 2
counts = Counter()
splitreviews = []

# Preprocessing dataset removal of punctuations and making the text lowercase
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

vocabsize = len(counts.keys())
hiddenSize = K
W1 = np.random.uniform(-1, 1,(vocabsize, hiddenSize))
W2 = np.random.uniform(-1, 1,(hiddenSize, vocabsize))

# Sigmoid Function
def sigmoid(v, derivative):
    if (derivative == True):
        return v * (1 - v)
    return 1/(1 + np.exp(-v))

# Softmax function on a vector
def softmax(x):   
    output = np.exp(x) / np.sum(np.exp(x))
    return output

# FeedForward Neural Network implemented from scratch
def feedForward(X):
    z = np.dot(X, W1)
    z = z/CONTEXT_SIZE 
    z2 = sigmoid(z,False) 
    z3 = np.dot(z2, W2)
    output = softmax(z3)
    return output,z2

# Back Propogation Algorithm for the feed forward neural network
def backprop(X, y, output,z2,W1,W2):
    output_error = y - output
    output_delta = output_error * sigmoid(output,True)
    
    z2_error = output_delta.dot(W2.T) 
    z2_delta = z2_error * sigmoid(z2,True) 
    
    A1 = np.asmatrix(X)
    z2_delta = np.asmatrix(z2_delta)
    W1 += A1.T.dot(z2_delta)

    A2 = np.asmatrix(z2)
    output_delta = np.asmatrix(output_delta)
    W2 += A2.T.dot(output_delta)
    return W1,W2,output_error,z2_error

# Function for One hot encoding for a given word.
def onehot(word):
    encod = np.zeros(vocabsize)
    encod[counts[word]] = 1
    return encod

for i in range(5):
    print(i)
    for review in splitreviews:
        avg_error = np.zeros(vocabsize)
        avg_error_z2 =  np.zeros(K)
        for i in range(1,len(review)-1):
            y = onehot(review[i])
            X = onehot(review[i-1]) + onehot(review[i+1]) 
            output,z2 = feedForward(X)
            W1,W2,output_error,z2_error = backprop(X, y, output,z2,W1,W2)
            avg_error += output_error ** 2
            avg_error_z2 += z2_error ** 2
    print(np.average(avg_error))
    print(np.average(avg_error_z2))

word_embeddings = {}
for word in vocabulary:
    word_embeddings[word] = np.dot(onehot(word),W1)

# Function to find the top 10 most similar words for a given word
def find_word_embeddings(searchword):
    top = []

    for i in range(10):
        top.append([0," "])

    for word in vocabulary:
        a = word_embeddings[searchword]
        b = word_embeddings[word]
        cos_sim = np.dot(a, b)/(norm(a)*norm(b))
        index = 0
        for item in top:
            if cos_sim > item[0] and word != searchword:
                top.insert(index,[cos_sim,word])
                top.pop(10)
                break
            index += 1
    #print(top)
    return top

top = find_word_embeddings("camera")
print(top)

word_embeddings = {}
for word in vocabulary:
    word_embeddings[word] = np.dot(onehot(word),W1)

keys = ['camera', 'product', 'good', 'strong', 'look']

# Code for tsne plots for word embeddings for words similar to ['camera', 'product', 'good', 'strong', 'look']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word in find_word_embeddings(word):
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
    plt.savefig("model2.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()

tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)

# Pre-trained embeddings 
wv = api.load('word2vec-google-news-300')
print(wv.most_similar("camera", topn=10))