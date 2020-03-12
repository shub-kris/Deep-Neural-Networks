from nltk.corpus import reuters
from nltk import bigrams, trigrams
from nltk.util import ngrams
from collections import Counter, defaultdict
import random

# demo_sent = reuters.sents()[0]
# print(demo_sent)
# # print(list(bigrams(demo_sent)))
# print(list(trigrams(demo_sent)))

'''
defaultdict means that if a key is not found in the dictionary, 
then instead of a KeyError being thrown, a new entry is created.
Lambda is initializing the default value to be zero and a dictionary
'''

# trigram_model = defaultdict(lambda: defaultdict(lambda: 0))  # A dictionary of dictionary
teragram_model = defaultdict(lambda: defaultdict(lambda: 0))

'''
Create the co-occurence table
'''

# for sent in reuters.sents():
    # for w1, w2, w3 in trigrams(sent, pad_right = True, pad_left = True):
        # trigram_model[(w1, w2)][w3] += 1

for sent in reuters.sents():
    for w1, w2, w3, w4 in ngrams(sent, 4, pad_right = True, pad_left = True):
        teragram_model[(w1, w2, w3)][w4] += 1

'''
Convert counts into probabilities
'''

# for ww in trigram_model:
    # count = float(sum(trigram_model[ww].values()))
    # for w in trigram_model[ww]:
        # trigram_model[ww][w] /= count

for ww in teragram_model:
    count = float(sum(teragram_model[ww].values()))
    for w in teragram_model[ww]:
        teragram_model[ww][w] /= count           

# print(trigram_model["today","the"])
# exit()        
        
'''
Text Generator
'''
text = [None, None, None]   # Starting words
end_cond = False

while not end_cond:
    r = random.random()  # Random number b/w 0 and 1
    threshold = .0
    # print('Hello')
    # print(trigram_model[tuple(text[-2:])].keys())
    for word in teragram_model[tuple(text[-3:])].keys():
        threshold += teragram_model[tuple(text[-3:])][word]
        if threshold >= r:
            text.append(word)
            break
            
    if text[-3:] == [None, None, None]:
        end_cond = True

print(' '.join([i for i in text if i]))