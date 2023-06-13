import pandas as pd
import csv
import re
import json

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

def generate_freq_dic(token_list):
    '''
    Input: list of tokens
    return: a dictionary which includes the tokens and their frequency
    '''
    current_freq = {}
    #calculate term frequency
    for i in token_list:
        if i == " ":
            continue
        if i == "" or i == '':
            continue
        try: 
            current_freq[i] += 1
        except:
            current_freq[i] = 1

    return current_freq


#import the vocab
df = pd.read_csv(r'vocabulary_1.csv')
#print(df["word"].head())

#convert word column into a list
vocab_list = df.word.values.tolist()
#print(vocab_list[:100])
print("Number of words", len(vocab_list))

#create the inverted index dictionary
vocab_dict = {}
for key in vocab_list:
    vocab_dict[key] = []

#a list of all the documents
list_doc_str = []
columns = ['qid','pid','query','passage']
passa = pd.read_csv('./coursework-1-data/candidate-passages-top1000.tsv', sep='\t', header=None,names=columns)
#subset the relevant columns
passa = passa[['pid', 'passage']]
#remove duplicate rows (i.e. keep unique instances od pid and passage)
passa = passa.drop_duplicates()

#convert these columns to list so its if faster to loop through them
pid = passa['pid'].tolist()
list_doc_str = passa['passage'].to_list()
total_passages = len(pid)

#loop through each document and update the inverted index
for i in range(total_passages):

    #preprocess the passage
    current = list_doc_str[i]
    tokens = preprocessing_text(current)

    #create a frequency dictionary
    term_freq = generate_freq_dic(tokens)
    
    #update the inverted index
    for key in term_freq: 
        try: 
            vocab_dict[key].append([pid[i], term_freq[key]])
        except:
            continue

    
#save as a json file
json = json.dumps(vocab_dict)
f = open("inverted_index.json","w")
f.write(json)
f.close()






