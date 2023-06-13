#import relevant libraries
import re
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import csv
import pandas as pd

#preprocessing function
def preprocessing_text(text):
    '''
    Input: a whole passage as a string (type: str)
    Pre-process the text to remove puncuations and turn to lower case
    After that convert the text into a list of tokens (tokenization)
    Return: list of tokens
    '''
    text = text.replace('\n', " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    tokens = text.split(" ") #split based on spaces
    return tokens

#open and save the data from the file
with open('./coursework-1-data/passage-collection.txt', 'r') as file:
    data = file.read().replace('\t', ' ').replace('\n', " ")

#Preprocessing steps
tokens = preprocessing_text(data)
tokens.sort() #sort the tokens

#Creating a vocabulary (unique terms) and term frequency
vocabulary = {}

#calculate term frequency
for i in tokens:
    if i == " ":
        continue
    if i == "" or i == '':
        continue
    try: 
        vocabulary[i] += 1
    except:
        vocabulary[i] = 1


total_words = len(tokens)
print("Total vocabulary is", len(vocabulary))
stop_word_included = vocabulary.copy()

#removing the stop words from the vocabulary (I think we should remove them given the way our IR works)
stopwords = stopwords.words('english')
print("Total stopwords", len(stopwords))
#print(stopwords)
for stop_word in stopwords:
    vocabulary.pop(stop_word, None)

print("Total vocabulary without stopwords is", len(vocabulary))

#save the vocabulary as a csv file (this is without stop-words)
with open('vocabulary.csv', 'w') as f:
    for key in vocabulary.keys():
        f.write("%s,%s\n"%(key,vocabulary[key]))


print("Stopword included", len(stop_word_included))
#with stop-words
norm_freq_s = {}
#calculate normalized frequency
for key in stop_word_included:
   norm_freq_s[key] = stop_word_included[key] / total_words

print("Total with stopwords", len(norm_freq_s))

#without stop-words
norm_freq = {}
total_words = sum(vocabulary.values())
#calculate normalized frequency
for key in vocabulary:
   norm_freq[key] = vocabulary[key] / total_words

print("Total without stopwords", len(norm_freq))


#sort the dictionary based on normalized frequency
sorted_nrom_freq_s = sorted(norm_freq_s.items(), key=lambda x:x[1], reverse=True)
sorted_nrom_freq_s = dict(sorted_nrom_freq_s) #convert to dictionary

#sort the dictionary based on normalized frequency
sorted_nrom_freq = sorted(norm_freq.items(), key=lambda x:x[1], reverse=True)
sorted_nrom_freq = dict(sorted_nrom_freq) #convert to dictionary

#set the ranking for each terms based on the normalized frequency
ranking_s = {}
rank = 1
for key in sorted_nrom_freq_s:
    ranking_s[key] = rank
    rank += 1

#set the ranking for each terms based on the normalized frequency
ranking = {}
rank = 1
for key in sorted_nrom_freq:
    ranking[key] = rank
    rank += 1


#plot zip's law without stop words
term_freq_rank = np.array(list(ranking.values()))
prob_occurance = np.array(list(sorted_nrom_freq.values()))


N = len(ranking)
sum_n = 0
for n in range(1,N+1):
    sum_n += n ** (-1)
zip_freq = []
for i in range(N):
    freq = 1 / (term_freq_rank[i] * sum_n)
    zip_freq.append(freq)

plt.plot(term_freq_rank, zip_freq, linestyle='dashed')
plt.plot(term_freq_rank, prob_occurance)
classes = ['theory (Zipf\'s Law)', 'data']
plt.legend(labels=classes)
plt.xlabel('Term frequency ranking', fontsize=10)
plt.ylabel('Term prob. of occurance', fontsize=10)
plt.title("Zipf's Law vs. Data (stopwords excluded)")
#plt.show()
plt.savefig('zipf_no_stopwords.pdf')

plt.clf()
plt.plot(term_freq_rank, zip_freq, linestyle='dashed')
plt.plot(term_freq_rank, prob_occurance)
classes = ['theory (Zipf\'s Law)', 'data']
plt.legend(labels=classes)
plt.xlabel('Term frequency ranking (log)', fontsize=10)
plt.ylabel('Term prob. of occurance (log)', fontsize=10)
plt.title("Zipf's Law vs. Data (stopwords excluded)")
plt.yscale('log')
plt.xscale('log')
#plt.show()
plt.savefig('zipf_no_stopwords_log.pdf')

#save the table that contains the vocabulary, term frequancy, normalized freq, and ranking as a pandas file
vocab = list(sorted_nrom_freq.keys())
norm_freq = list(sorted_nrom_freq.values())
ranks = list(ranking.values())

vocab_csv = {'word': vocab, 'norm_freq': norm_freq, "rank": ranks}  

df = pd.DataFrame(vocab_csv) 
    
# saving the dataframe 
df.to_csv('vocabulary_1.csv')


#plot zip's law with stopwords
term_freq_rank = np.array(list(ranking_s.values()))
prob_occurance = np.array(list(sorted_nrom_freq_s.values()))

N = len(ranking_s)
sum_n = 0
for n in range(1,N+1):
    sum_n += n ** (-1)
zip_freq = []
for i in range(N):
    freq = 1 / (term_freq_rank[i] * sum_n)
    zip_freq.append(freq)

plt.clf()
plt.plot(term_freq_rank, zip_freq, linestyle='dashed')
plt.plot(term_freq_rank, prob_occurance)
classes = ['theory (Zipf\'s Law)', 'data']
plt.legend(labels=classes)
plt.xlabel('Term frequency ranking', fontsize=10)
plt.ylabel('Term prob. of occurance', fontsize=10)
plt.title("Zipf's Law vs. Data (stopwords included)")
#plt.show()
plt.savefig('zipf_stopwords.pdf')


plt.clf()
plt.plot(term_freq_rank, zip_freq, linestyle='dashed')
plt.plot(term_freq_rank, prob_occurance)
classes = ['theory (Zipf\'s Law)', 'data']
plt.legend(labels=classes)
plt.xlabel('Term frequency ranking (log)', fontsize=10)
plt.ylabel('Term prob. of occurance (log)', fontsize=10)
plt.title("Zipf's Law vs. Data (stopwords included)")
plt.yscale('log')
plt.xscale('log')
#plt.show()
plt.savefig('zipf_stopwords_log.pdf')


