import numpy, json, argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
numpy.random.seed(1337)

# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
	print('Reading in data from {0}...'.format(corpus_file))
	words = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			words.append(parts[0])
			if binary_classes:
				if parts[1] in ['GPE', 'LOC']:
					labels.append('LOCATION')
				else:
					labels.append('NON-LOCATION')
			else:
				labels.append(parts[1])	
	print('Done!')
	return words, labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
	print('Reading in embeddings from {0}...'.format(embeddings_file))
	embeddings = json.load(open(embeddings_file, 'r'))
	embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
	print('Done!')
	return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
	vectorized_words = []
	for word in words:
		try:
			vectorized_words.append(embeddings[word.lower()])
		except KeyError:
			vectorized_words.append(embeddings['UNK'])
	return numpy.array(vectorized_words)
   
if __name__ == '__main__':
	print('\n\nSTART SCIRPT\n')
	
	parser = argparse.ArgumentParser(description='KerasNN parameters')
	parser.add_argument('data', metavar='named_entity_data.txt', type=str, help='File containing named entity data.')
	parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
	parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
	args = parser.parse_args()
	
	# Read in the data and embeddings
	X, Y = read_corpus(args.data, binary_classes = args.binary)
	embeddings = read_embeddings(args.embeddings)

	# print('Writing words to available.txt')
	# unique_words = [word.lower() for word in set(X)]
	# print("Unique's = ", unique_words[:20])
	# print("Total unique words :", len(unique_words))
	# print("Total embeddings:", len(embeddings.keys()))
	# words = [word for word in embeddings.keys() if word.lower() not in unique_words]
	# print("Total available words:", len(words))
	# f = open("available.txt", "wb+")
	# f.write('\n'.join(words).encode('utf-8'))
	# f.close()
	# print('Done!')

	# Transform words to embeddings
	X = vectorizer(X, embeddings)
	
	# Transform string labels to one-hot encodings
	encoder = LabelBinarizer()
	Y = encoder.fit_transform(Y) # Use encoder.classes_ to find mapping of one-hot indices to string labels
	if args.binary:
		Y = numpy.where(Y == 1, [0,1], [1,0])
	
	# Split in training and test data
	split_point = int(0.75*len(X))
	Xtrain = X[:split_point]
	Ytrain = Y[:split_point]
	Xtest = X[split_point:]
	Ytest = Y[split_point:]
	
	# Define the properties of the perceptron model
	model = Sequential()
	model.add(Dense(input_dim=X.shape[1], output_dim=Y.shape[1]))
	# model.add(Dense(input_dim=X.shape[1], units=Y.shape[1])) # Original, modified because older version
	model.add(Activation("linear"))
	sgd = SGD(lr=0.01)
	loss_function = 'mean_squared_error'
	model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])

	# Train the perceptron
	model.fit(Xtrain, Ytrain, verbose=1, nb_epoch=25, batch_size=32)
	# model.fit(Xtrain, Ytrain, verbose=1, epochs=1, batch_size=32) # Original, modified because older version

	# Get predictions
	Yguess = model.predict(Xtest)
	
	# Convert to numerical labels to get scores with sklearn in 6-way setting
	Yguess = numpy.argmax(Yguess, axis = 1)
	Ytest = numpy.argmax(Ytest, axis = 1)
	print('Classification accuracy on test: {0}'.format(accuracy_score(Ytest, Yguess)))
	# Breaux, Medicare, Harsco	ORG
	# September, tomorrow, 1929	DATE
	# Bush, Forrest, GORE		PERSON
	# Earth ,europe, Siberia	LOC
	# Bremen, pakistan, moscow	GPE
	# two, one, 17		CARDINAL

	X_words_og = ["vvd", "mountain", "ronaldo", "two-hundred", "west-germany", "1533", "120mm"]
	expectation = ["org", "loc", "person", "cardinal", "gpe", "date", "cardinal"]
	X_words = vectorizer(X_words_og, embeddings)
	Y_words = model.predict(X_words)
	Y_words = numpy.argmax(Y_words, axis=1)
	print(list(zip(X_words_og, Y_words)))
	print("BINARY: {0=LOCATION, 1=NON-LOCATION}")
	print("MULTICLASS: {0=CARDINAL, 1=DATE ,2=GPE, 3=LOC, 4=ORG, 5=PERSON}")
	# GPE = geo - political entity
	# LOC = location
	# PERSON = person
	# ORG = organisation
	# DATA = date
	# CARDINAL = cardinal number
	# number(CARDINAL)
	print('\nEND OF SCIRPT')

# f = open("demofile2.txt", "a")
# f.write("Now the file has more content!")
# f.close()

# geo-political entity (GPE)
# location (LOC)
# person (PERSON)
# organisation (ORG)
# date (DATE)
# cardinal number (CARDINAL


