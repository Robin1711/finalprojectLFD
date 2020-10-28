from main import *
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import Counter
import nltk
import re
import string
import json

"""
scp /home/nvtslot/Documents/Master/"Learning From Data"/finalprojectLFD-main/articlebodies.txt s2536420@karora.let.rug.nl:/home/s2536420/word2vec/articlebodies.txt
"""

def body_text(cop_selection=None):
	f1 = open("articlebodies.txt", "w+")
	cop_data = read_data(cop_selection)
	articles = []
	stop_words = nltk.corpus.stopwords.words("english")
	stopwords_dict = Counter(stop_words)

	for cop in cop_data:
		articles = cop_data[cop]['articles']
		for article in articles:
			article_text = article["body"].lower().replace("\n", " ")
			article_text = " ".join([word for word in article_text.split() if word not in stopwords_dict])
			
			#f1.write(article_text.translate(str.maketrans("", "", string.punctuation)))
			f1.write(article_text)
		print("COP{} body converted".format(cop))
	f1.close()

def main():
	body_text()

	vocabulary = []
	f2 = open("articlebodies.txt", "r")
	textbody = f2.read()
	
	sentences = nltk.tokenize.sent_tokenize(textbody)
	print("Sentence tokenization complete")
	for sentence in sentences:
		words = nltk.tokenize.word_tokenize(sentence)
		vocabulary.append(words)

	print("Word tokenization complete")

	model = Word2Vec(vocabulary, size=100, window=3, min_count=15, workers=4)
	print("Model trained")
	model.wv.save_word2vec_format("gensim_embeddings.txt", binary=False)
	f2.close()
	print("Model saved")

	f3 = open("gensim_embeddings.txt", "r")
	v = {"vectors": {}}
	for line in f3:
		try:
			w, n = line.split(" ", 1)
			v["vectors"][w] = list(map(float, n.split()))
		except ValueError:
			pass
	v = v["vectors"]

	print("Converted to JSON")
	with open("word_embeddings.txt"[:-4] + ".json", "w") as out:
		json.dump(v, out)

	f3.close()

if __name__ == "__main__":
	main()