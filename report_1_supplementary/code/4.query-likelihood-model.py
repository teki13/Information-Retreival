#import relevant libraries
import json
import pandas as pd
import math
import numpy as np
import re
from nltk.corpus import stopwords

#Functions
def preprocessing_text(text):
    '''
    Input: a whole passage as a string (type: str)
    Pre-process the text to remove puncuations and turn to lower case
    After that convert the text into a list of tokens (tokenization)
    Return: list of tokens
    '''
    text = text.replace('\n', " ")
    text = re.sub(r'[^\w\s]', '', text)
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
        try: 
            current_freq[i] += 1
        except:
            current_freq[i] = 1

    return current_freq


#load the inverted index dictionary into memory
with open('inverted_index.json') as json_file:
    inverted_index = json.load(json_file)

#create a column with unique passage ids. 
#a list of all the documents
list_doc_str = []
columns = ['qid','pid','query','passage']
passa = pd.read_csv('./coursework-1-data/candidate-passages-top1000.tsv', sep='\t', header=None,names=columns)
all_c = passa
#subset the relevant columns
passa = passa[['pid']]
#remove duplicate rows (i.e. keep unique instances od pid and passage)
passa = passa.drop_duplicates()
pid = passa['pid'].tolist()
#subset the qid and pid pairs
qid_pid = all_c[['pid', 'qid']]


#Compute term frequencies ofr the documents


frames = []
#create dataset by pid and by word
for i in inverted_index:
    temp = pid
    list_id = []
    term_freq = []
    for j in inverted_index[i]:
        list_id.append(j[0])
        term_freq.append(int(j[1]))

    #create a dataframe
    d = {'pid': list_id, 'tf_p': term_freq}
    df = pd.DataFrame(data=d)
    df['term_name_p'] = i
    
    #append all frames
    frames.append(df)
    
result = pd.concat(frames)

#compute the idf for each of the words

N = len(pid) #total number of passages
word = []
appearances = []

#loop through each word
for i in inverted_index:
    
    word.append(i)
    n = len(inverted_index[i])
    appearances.append(n)
    
idf = {'term': word, 'n': appearances}
idf_df = pd.DataFrame(data=idf)
idf_df['N'] = N


#we can now combine the tables to calculate tf-idf for the documents
tf_idf_doc = pd.merge(result,idf_df,left_on='term_name_p',right_on='term', how='left')



#Compute term frequencies for the queries
# import the queries
columns = ['qid','query']
queries = pd.read_csv('./coursework-1-data/test-queries.tsv', sep='\t', header=None,names=columns)

#let's find term frequency for each of the queries
qid = queries['qid'].tolist()
query_list = queries['query'].tolist()

frames = []

for i in range(len(qid)):
    
    #tokenize the query
    tokenize = preprocessing_text(query_list[i])
    
    #term freq count 
    term_freq = generate_freq_dic(tokenize)

    #remove stop-words
    stopwords_n = stopwords.words('english')
    
    for stop_word in stopwords_n:
        term_freq.pop(stop_word, None)
    
    terms = list(term_freq.keys())
    term_f = list(term_freq.values())
    
    #make a dataframe
    d = {'term_q': terms, 'tf_q': term_f}
    df = pd.DataFrame(data=d)
    df['qid'] = qid[i]
    
    #append to list of dataframes
    frames.append(df)

#concatanate the results for all the queries
all_queries = pd.concat(frames)

#merge the queries with the associated pid
all_queries = pd.merge(all_queries,qid_pid,left_on='qid',right_on='qid', how='left')



####### Laplace Smoothing #############
#merge query with paragraphs
all_q_d = pd.merge(all_queries,tf_idf_doc,left_on=['pid', 'term_q'],right_on=['pid', 'term_name_p'], how='left')

#calculate document length for each of the documents
document_length = all_c[['pid', 'passage']]
document_length = document_length.drop_duplicates() #drop the duplicates

pid_list = []
doc_length = []

#loop through each row and tokenize each of the passages
for index, row in document_length.iterrows():
    pid_list.append(row['pid'])
    
    #pre-process the passage and remove stop-words
    pre_process = preprocessing_text(row['passage'])
    
    stopwords_n = stopwords.words('english')
    
    for stop_word in stopwords_n:
        try:
            pre_process.remove(stop_word)
        except:
            continue
    
    doc_length.append(len(pre_process))

iterim = {'pid': pid_list, 'dl': doc_length}
doc_len = pd.DataFrame(data=iterim)

#replace the nan with 0s. This would happen when we merge and a document word was not present in the query
all_q_d["tf_p"] = all_q_d["tf_p"].fillna(0)
#add one to each document term frequency
all_q_d['smooth'] = all_q_d['tf_p'] + 1
#add vocabulary length
V = len(inverted_index)
all_q_d['V'] = V
#concatenate the document length for each of the documents
all_q_d = pd.merge(all_q_d,doc_len,left_on=['pid'],right_on=['pid'], how='left')
#calculate the probability
all_q_d['P'] = all_q_d['smooth'] / (all_q_d['V'] + all_q_d['dl'])
#take the logarithm of the probability
all_q_d['log(P)'] = np.log(all_q_d['P'])

#for each of the queries take the product of the probability (sum of log probability)
all_q_d = all_q_d.groupby(['qid', 'pid']).agg({'log(P)':'sum', 'P':'prod'}).reset_index()

#sort results in descending order
final_sum_all = all_q_d.sort_values(by=['qid', 'P'], ascending=False)

#select the top 100 paragraphs per query
top_100_par = final_sum_all.groupby(['qid']).head(100)

#join on the qid to get the same order as the one in the file
final_ranking = pd.merge(queries,top_100_par,left_on='qid',right_on='qid', how='left')

#save the file as a csv file
#subset columns
final_file = final_ranking[["qid", "pid", "log(P)"]]
final_file = final_file.rename(columns={"log(P": "score"})

print("The length of the laplace smoothing file is", len(final_file))

#save the file as a csv file
final_file.to_csv("laplace.csv", header = False)  






########## Lidstone Correction #############

#merge query with paragraphs
all_q_d = pd.merge(all_queries,tf_idf_doc,left_on=['pid', 'term_q'],right_on=['pid', 'term_name_p'], how='left')

epsilon = 0.1
#replace the nan with 0s
all_q_d["tf_p"] = all_q_d["tf_p"].fillna(0)
#add epsilon to each document term frequency
all_q_d['smooth'] = all_q_d['tf_p'] + epsilon
#add vocabulary length
V = len(inverted_index)
all_q_d['V'] = V
#concatenate the document length for each of the documents
all_q_d = pd.merge(all_q_d,doc_len,left_on=['pid'],right_on=['pid'], how='left')
#calculate the probability
all_q_d['P'] = all_q_d['smooth'] / (all_q_d['V'] + (epsilon * all_q_d['dl']))
#take the logarithm of the probability
all_q_d['log(P)'] = np.log(all_q_d['P'])

#for each of the queries take the product of the probability (sum of log probability)
all_q_d = all_q_d.groupby(['qid', 'pid']).agg({'log(P)':'sum', 'P':'prod'}).reset_index()

#sort results in descending order
final_sum_all = all_q_d.sort_values(by=['qid', 'P'], ascending=False)

#select the top 100 paragraphs per query
top_100_par = final_sum_all.groupby(['qid']).head(100)

#join on the qid to get the same order as the one in the file
final_ranking = pd.merge(queries,top_100_par,left_on='qid',right_on='qid', how='left')

#subset columns
final_file = final_ranking[["qid", "pid", "log(P)"]]
final_file = final_file.rename(columns={"log(P": "score"})

print("The length of the lidstone smoothing file is", len(final_file))

#save the file as a csv file
final_file.to_csv("lidstone.csv", header = False)  





######### Dirichlet Smoothing ###################
#merge query with paragraphs
all_q_d = pd.merge(all_queries,tf_idf_doc,left_on=['pid', 'term_q'],right_on=['pid', 'term_name_p'], how='left')

#calculate how many times each word appeard in the entire corpus
word = []
count = []

#loop through each word
for i in inverted_index:
    word.append(i)
    sum_f = 0
    for j in inverted_index[i]:
        sum_f += j[1]
    count.append(sum_f)

#create a dataframe with column for term and a column for the total appearances of the word in the corpus
iterim = {'term': word, 'cor_f': count}
cor_f = pd.DataFrame(data=iterim)


all_q_d['mu'] = 50 #given in coursework
#replace the nan with 0s
all_q_d["tf_p"] = all_q_d["tf_p"].fillna(0)
#concatenate the document length for each of the documents
all_q_d = pd.merge(all_q_d,doc_len,left_on=['pid'],right_on=['pid'], how='left')
#concatenate the number of times the word appeard in the entire corpus
all_q_d = pd.merge(all_q_d,cor_f,left_on=['term_q'],right_on=['term'], how='left')
#add a column for the entire corpus length
all_q_d['c_words'] = cor_f['cor_f'].sum()


#compute document part of equation = N / (N + mu) * language model
all_q_d['doc'] = (all_q_d['dl'] / (all_q_d['dl'] + all_q_d['mu'])) * (all_q_d['tf_p'] / all_q_d['dl'])
#compute corpus part of the equation = mu / (N + mu) * collection
all_q_d['col'] = (all_q_d['mu'] / (all_q_d['mu'] + all_q_d['dl'])) * (all_q_d['cor_f'] / all_q_d['c_words'])
#sum the two terms
all_q_d['di_s'] = np.log(all_q_d['doc'] + all_q_d['col'])

#for each of the queries take the product of the probability (sum of log probability)
all_q_d = all_q_d.groupby(['qid', 'pid']).agg({'di_s':'sum'}).reset_index()
#sort results in descending order
final_sum_all = all_q_d.sort_values(by=['qid', 'di_s'], ascending=False)
#select the top 100 paragraphs per query
top_100_par = final_sum_all.groupby(['qid']).head(100)
#join on the qid to get the same order as the one in the file
final_ranking = pd.merge(queries,top_100_par,left_on='qid',right_on='qid', how='left')

#save the file as a csv file
#subset columns
final_file = final_ranking[["qid", "pid", "di_s"]]
final_file = final_file.rename(columns={"di_s": "score"})

print("The length of the dirichlet smoothing file is", len(final_file))

#save the file as a csv file
final_file.to_csv("dirichlet.csv", header = False)  

