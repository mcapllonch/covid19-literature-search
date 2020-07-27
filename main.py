import os
from collections import OrderedDict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import config as cfg



def find_kw(kw, text):
	"""
	Find a keyword 'kw' in 'text' and return its number of occurences.
	kw must be lowered.
	"""

	# Count
	count = text.count(kw)
	return count

def prepare_stopwords():
	""" Prepare the set of words to be stopped in English. """
	cfg.stopwords = set(stopwords.words('english'))

def count_words(text):
	""" Prepare a piece of text (a string): lower, stopwords, and return dictionary of words with frequency of occurence. """

	# Lower the text
	text = text.lower()
	# Tokenize words
	words = word_tokenize(text)
	# Stop words
	words_ = [w for w in words if w not in cfg.stopwords]
	# Set of words
	wordset = set(sorted(words_))

	# Dictionary of words
	words = OrderedDict()
	for w in wordset:
		words[w] = words_.count(w)
	
	return words

def search_kw(keywords, min_freq=0):
	"""
	Search for keywords in all the abstracts of the metadata.
	Iterate over all abstracts and yield a dictionary of occurences as {cord_uid: occurences}.
	"""

	occurences = {}

	# Iterate over all abstracts
	i = 0
	for cord_uid, abstract in zip(metadata.cord_uid, metadata.abstract):
		if i % 100 == 0:
			print(i)
		# Get the words in the abstract
		words = count_words(abstract)
		# Get the number of occurences of the word in this abstract
		for kw in keywords:
			# Look for it in the abstract's words
			try:
				n = words[kw]
			except KeyError:
				pass
			else:
				# Add it to the counter
				try:
					occurences[cord_uid] += n
				except KeyError:
					occurences[cord_uid] = n
		i += 1

	# occurences to DataFrame
	occurences = pd.DataFrame({'cord_uid': list(occurences.keys()), 'frequency': list(occurences.values())})

	# Filter using min_freq
	occurences = occurences[occurences.frequency >= min_freq]

	# Sort by frequency in descending order
	occurences.sort_values(by='frequency', ascending=False, inplace=True, ignore_index=True)

	return occurences

# Define folder paths
cfg.cwd = os.getcwd()
cfg.folders = {'corddata': os.path.join(cfg.cwd, 'corddata')}

# Prepare the stopwords list
prepare_stopwords()

# Read meta-data
# Select only a small fraction of the data
metadata = pd.read_csv(os.path.join(cfg.folders['corddata'], 'metadata.csv')).loc[::100]

print('Full metadata file:')
print(metadata.shape[0])

# Filter null abstracts
metadata.dropna(subset=['abstract'], inplace=True)
print('After removing null abstracts:')
print(metadata.shape[0])

# Filter null texts
has_full_text = ~(pd.isna(metadata.pdf_json_files) & pd.isna(metadata.pmc_json_files))
metadata = metadata.loc[has_full_text].reset_index()
print('After removing null full texts:')
print(metadata.shape[0])

# Select keywords to search for
kw = ['rt', 'pcr', 'polymerase', 'chain']

# Search in the whole database
occurences = search_kw(kw, min_freq=2)

print(occurences.head())

# Nice display of the results
results = pd.merge(occurences, metadata, how='left', left_on='cord_uid', right_on='cord_uid')
results = results[['frequency', 'title', 'publish_time', 'url', 'abstract']]

print(results.head())