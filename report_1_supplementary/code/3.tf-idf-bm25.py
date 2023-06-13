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


#Compute tf-idf for the documents
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

#convert the term frequancy to log space frequency
result['tf_log_p'] = np.log10(1 + result['tf_p'])

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
idf_df['IDF'] = np.log10(idf_df['N']/idf_df['n'])

#we can now combine the tables to calculate tf-idf for the documents
tf_idf_doc = pd.merge(result,idf_df,left_on='term_name_p',right_on='term', how='left')
tf_idf_doc['tf_idf_p'] = tf_idf_doc['tf_log_p'] * tf_idf_doc['IDF']

#for each of the documents compute the normalizing term
tf_idf_doc['norm'] = tf_idf_doc['tf_idf_p'] * tf_idf_doc['tf_idf_p']
#sum norm by document
df_norm = tf_idf_doc.groupby('pid').agg({'norm':'sum'}).reset_index()
df_norm['norm_d'] = np.sqrt(df_norm['norm'])
df_norm = df_norm.drop(['norm'], axis=1)

#join with the qid for which the pid is relevant for
tf_idf_doc = pd.merge(tf_idf_doc,qid_pid,left_on='pid',right_on='pid', how='left')



#Compute tf-idf for the queries
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
#calculate log term frequency
all_queries['tf_log_q'] = np.log10(1 + all_queries['tf_q'])

#combine the queries dataset with the idf dataset to calculate tf-idf
tf_idf_q = pd.merge(all_queries,idf_df,left_on='term_q',right_on='term', how='left')
tf_idf_q['tf-idf_q'] = tf_idf_q['tf_log_q'] * tf_idf_q['IDF']

#calculate the query norm

#for each of the queries compute the normalizing term
tf_idf_q['norm'] = tf_idf_q['tf-idf_q'] * tf_idf_q['tf-idf_q']

#sum norm by document
df_norm_q = tf_idf_q.groupby('qid').agg({'norm':'sum'}).reset_index()
df_norm_q['norm_q'] = np.sqrt(df_norm_q['norm'])
df_norm_q = df_norm_q.drop(['norm'], axis=1)




#Rank the queries
#merge the paragraph table with the queries 
all_q_d = pd.merge(tf_idf_q,tf_idf_doc,left_on=['qid','term_q'],right_on=['qid', 'term'], how='outer')

#replace the nan with 0s - this is when a word does not match between query and a passage
all_q_d["tf_p"] = all_q_d["tf_p"].fillna(0)
all_q_d["tf-idf_q"] = all_q_d["tf-idf_q"].fillna(0)

#multiply the 2 tf-idf of the query with the tf-idf of the paragraph
all_q_d['tf_idf_pord'] = all_q_d['tf-idf_q'] * all_q_d['tf_idf_p']

#group by pid and get the sum
final_sum_all = all_q_d.groupby(['qid', 'pid']).agg({'tf_idf_pord':'sum'}).reset_index()

#merge with normalizing terms
final_sum_all = pd.merge(final_sum_all,df_norm,left_on='pid',right_on='pid', how='left')
final_sum_all = pd.merge(final_sum_all,df_norm_q,left_on='qid',right_on='qid', how='left')

#calculate cosine similarity
final_sum_all['cosine_similarity'] = final_sum_all['tf_idf_pord'] / (final_sum_all['norm_d'] * final_sum_all['norm_q'])

#sort results in descending order
final_sum_all = final_sum_all.sort_values(by=['qid', 'cosine_similarity'], ascending=False)

#select the top 100 paragraphs per query
top_100_par = final_sum_all.groupby(['qid']).head(100)

#join on the qid to get the same order as the one in the file provided
final_ranking = pd.merge(queries,top_100_par,left_on='qid',right_on='qid', how='left')



#save the file as a csv file

#subset columns
final_file = final_ranking[["qid", "pid", "cosine_similarity"]]
final_file = final_file.rename(columns={"cosine_similarity": "score"})

print("The length of the final tf-idf file is", len(final_file))
#save the file as a csv file
final_file.to_csv("tfidf.csv", header = False)  






################# BM25 ######################

#assuming no relevance information
r = 0
R = 0

k_1 = 1.2
k_2 = 100
b = 0.75

#number of documents in the corpus (N) - total # of documents in the passages

#calculate document length for each of the passages
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

#calculate the average document length
avg_doc_len = doc_len['dl'].mean()
doc_len['avdl'] = avg_doc_len
doc_len['dl/avdl'] = doc_len['dl'] / doc_len['avdl']

#query and term frequency
tf_q = tf_idf_q[['term_q', 'tf_q', 'qid']]

#term frequency in the passages
tf_doc = tf_idf_doc[['pid', 'tf_p', 'term', 'qid']]

# count in how many documents does a term appear in
n_documents = tf_idf_doc[['pid', 'term_name_p']].drop_duplicates() #extract unique term-passage rows
n_documents = n_documents.groupby(['term_name_p'])['term_name_p'].count().reset_index(name='n')

#calculate the BM25 score for each query-document pair
bm_q_d = pd.merge(tf_q,tf_doc,left_on=['qid','term_q'],right_on=['qid', 'term'], how='outer')
bm_q_d["tf_q"] = bm_q_d["tf_q"].fillna(0)
bm_q_d["tf_p"] = bm_q_d["tf_p"].fillna(0)
bm_q_d["r"] = r
bm_q_d["R"] = R
bm_q_d["N"] = N
bm_q_d = pd.merge(bm_q_d,doc_len,left_on=['pid'],right_on=['pid'], how='left') #include information on document length
bm_q_d = pd.merge(bm_q_d,n_documents,left_on=['term'],right_on=['term_name_p'], how='left') #include information on how many documents contain a term
bm_q_d['K'] = k_1 * ((1-b)+ b * bm_q_d['dl/avdl'])
bm_q_d['part_1_score'] = ((bm_q_d['r'] + 0.5)/(bm_q_d['R'] - bm_q_d['r'] + 0.5)) / ((bm_q_d['n'] - bm_q_d['r'] +0.5)/(bm_q_d['N']- bm_q_d['n']- bm_q_d['R']+ bm_q_d['r'] + 0.5))
bm_q_d['part_2_score'] = ((k_1 + 1) * bm_q_d['tf_p']) / (bm_q_d['K'] + bm_q_d['tf_p'])
bm_q_d['part_3_score'] = ((k_2+1)*bm_q_d['tf_q'])/(k_2 + bm_q_d['tf_q'])
bm_q_d['bm25'] = np.log(bm_q_d['part_1_score']) * bm_q_d['part_2_score'] * bm_q_d['part_3_score']

#take the sum to obtain the final BM25 score
final_bm25 = bm_q_d.groupby(['qid', 'pid']).agg({'bm25':'sum'}).reset_index()

#sort results in descending order
final_bm25 = final_bm25.sort_values(by=['qid', 'bm25'], ascending=False)

#select the top 100 paragraphs per query
top_100_par = final_bm25.groupby(['qid']).head(100)

#join on the qid to get the same order as the one in the file
final_ranking = pd.merge(queries,top_100_par,left_on='qid',right_on='qid', how='left')

#save the file as a csv file

#subset columns
final_file = final_ranking[["qid", "pid", "bm25"]]
final_file = final_file.rename(columns={"bm25": "score"})

print("The length of the final BM25 is", len(final_file))

#save the file as a csv file
final_file.to_csv("bm25.csv", header = False)















