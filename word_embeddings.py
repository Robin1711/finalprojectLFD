# !/usr/bin/env python3
"""
word_embeddings.py

This script:
- saves the body of every article into a file
- performs preprocessing and tokenization
- Trains word embeddings
"""

from main import *
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import Counter
import nltk
import re
import os
import string
import json

def body_text(cop_selection=None, surpress_print=True):
	"""
	This module takes the textbody from every article in the dataset and preprocesses it.
	"""

	f1 = open("embeddings/articlebodies.txt", "w+")
	cop_data = read_data(cop_selection, surpress_print)
	articles = []

	# We create a stopwords dictionary to remove the stopwords from the articletext later on
	stop_words = nltk.corpus.stopwords.words("english")
	stopwords_dict = Counter(stop_words)

	# This loop adds every article body into a temporary plain text file
	for cop in cop_data:
		articles = cop_data[cop]['articles']
		for article in articles:
			article_text = article["body"].lower().replace("\n", " ")
			# Remove the stop words, we find them to not be informative for training word embeddings
			article_text = " ".join([word for word in article_text.split() if word not in stopwords_dict])
			f1.write(article_text)
	f1.close()

def tokenization():
	"""
	This module tokenizes the plain text from the articles utilizing NLTK's tokenizers.
	"""
	vocabulary = []
	f2 = open("embeddings/articlebodies.txt", "r")
	textbody = f2.read()
	
	# Tokenize the text into a list of sentences first
	sentences = nltk.tokenize.sent_tokenize(textbody)
	print("Sentence tokenization complete")
	for sentence in sentences:
		# Tokenize every sentence and add to a vocabulary list containing every word in the dataset
		words = nltk.tokenize.word_tokenize(sentence)
		vocabulary.append(words)

	print("Word tokenization complete")
	f2.close()
	return vocabulary

def train_embeddings():
	"""
	This module trains the word embeddings on the training set of COP 1-24.
	It then saves the embeddings as a .json file.
	"""
	vocabulary = tokenization()

	# Train and save the word2vec model
	model = Word2Vec(vocabulary, size=100, window=10, min_count=15, workers=4)
	print("Model trained")
	model.wv.save_word2vec_format("embeddings/word_embeddings.txt", binary=False)
	print("Model saved")

	# From the trained word2vec model, create a dictionary with the tokens/words as keys and vectors as values
	f3 = open("embeddings/word_embeddings.txt", "r")
	v = {"vectors": {}}
	for line in f3:
		# During training, we found that some lines in the document were empty. We skip those lines with try/except
		try:
			w, n = line.split(" ", 1)
			v["vectors"][w] = list(map(float, n.split()))
		except ValueError:
			pass
	v = v["vectors"]

	print("Converted to JSON")
	# Save the .json file containing the dictionary of word embeddings
	with open("embeddings/word_embeddings.txt"[:-4] + ".json", "w") as out:
		json.dump(v, out)
	f3.close()


def main():
	body_text()
	train_embeddings()

	# Remove the temporary plain text files
	os.remove("embeddings/articlebodies.txt")
	os.remove("embeddings/word_embeddings.txt")


if __name__ == "__main__":
	main()