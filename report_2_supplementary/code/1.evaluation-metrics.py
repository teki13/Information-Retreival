import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import csv
import pandas as pd


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


def calculate_mAP(k, validation_set, bm25_data):
    '''
    Function that calculates the mAP
    Inputs: 
        k - the percision rank at k
        validation_set - the original validation set (dataframe)
        bm25_data - the bm25 dataset with ranked passages for each query (dataframe)
        
    Outputs: 
        mAP - the mean average percision for top k passages (int)
    '''
    
    #select the top k passages for each query
    top_par = bm25_data.groupby(['qid']).head(k)
    
    #let's add a rank column
    top_par = top_par.copy()
    top_par.loc[:,'rank'] = top_par.groupby('qid')['bm25'].rank(method='dense', ascending=False)
    
    # we now need to perform a left join with the validation set 
    #in order to get the relevancy for each of the retreived passages

    #first subset the validation table
    validation_sub = validation_set[['qid', 'pid', 'relevancy']]

    #left join on bm25 table to obtain the relevancy column
    top_par = pd.merge(top_par, validation_sub, on=['qid','pid'], how = 'left')

    #calculate the cumulative sum over each relevancy
    top_par['cum_sum_rel'] = top_par.groupby('qid')['relevancy'].cumsum()
    
    #calculate the percision at each row
    top_par['average_percision'] = top_par['cum_sum_rel']/top_par['rank']
    
    # filter dataframe to only include rows where relevancy = 1
    top_par_filtered = top_par[top_par['relevancy'] == 1]
    
    # compute average percision
    avg_percision = top_par_filtered.groupby('qid')['average_percision'].agg(['sum', 'size'])['sum'] / top_par_filtered.groupby('qid')['average_percision'].agg(['sum', 'size'])['size']
    avg_percision = avg_percision.reset_index()
    
    #calculate the total number of unique queries in the validation set
    unique_q = validation_sub['qid'].unique()
    
    #calculate the mAP
    #mAP = avg_percision[0].sum() / len(avg_percision)
    mAP = avg_percision[0].sum() / len(unique_q)
    
    return mAP




def calculate_dcg(k, validation_set, bm25_set):
    '''
    A function that calculates the dcg score per query
    Inputs:
        k - the number of passages retreived (int)
        validation_set - the original validation set (dataframe)
        bm25_set - the ranked validation set using bm25 (dataframe)
        
    Outputs:
        ndcg - the dcg score per query (dataframe)
    '''
    
    top_par = bm25_set.groupby(['qid']).head(k)
    
    #include the rank and the relevancy score
    top_par = top_par.copy()
    #top_par.loc[:,'rank'] = top_par.groupby('qid')['bm25'].rank(method='dense', ascending=False)
    top_par['rank'] = top_par.groupby('qid').cumcount() + 1
    
    #first subset the validation table
    validation_sub = validation_set[['qid', 'pid', 'relevancy']]

    #left join on bm25 table to obtain the relevancy column
    top_par = pd.merge(top_par, validation_sub, on=['qid','pid'], how = 'left')
    
    #introduce the gain column and discounted gain columns
    top_par['gain'] = (2**top_par['relevancy']) - 1
    top_par['log'] = 1/ np.log2(top_par['rank'] + 1)
    top_par['discounted_gain'] = top_par['gain'] * top_par['log']
    
    #calculate ndcg
    ndcg = top_par.groupby('qid')['discounted_gain'].sum()
    ndcg = ndcg.reset_index()
    
    return ndcg


def calculate_ndcg(k, validation, final_bm25, validation_sorted):
    
    '''
    This function calclulates the ndcg score per query
    Inputs:
        k - the top k retrieved passages (int)
        validation - the original validation set (dataframe)
        final_bm25 - the ranked passages for each query using bm25
        validation_sorted - the original validation set sorted by relevance
    
    Output:
        total_ndcg - the ndcg score per query (dataframe)
    '''
    
    #calculate dcg per query
    ndcg_df = calculate_dcg(k, validation, final_bm25)
    
    #obtain ideal dcg score for top k
    top_k = validation_sorted.groupby(['qid']).head(k)
    top_k = top_k[['qid', 'pid']]
    
    #calculate the ideal dcg
    ideal_dcg = calculate_dcg(k, validation, top_k)
    ideal_dcg = ideal_dcg.rename(columns={"discounted_gain": "ideal_dcs"})
    
    
    #merge the ranked and ideal datasets by qid
    total_dcg = pd.merge(ndcg_df,ideal_dcg, on='qid', how='left')
    
    #calculate normalized dcg
    total_dcg['ndcg'] = total_dcg['discounted_gain'] / total_dcg['ideal_dcs']
    
    return total_dcg


#Creating a vocabulary
validation = pd.read_csv('./data/part2/validation_data.tsv',sep = '\t')
#subset the relevant columns
passa = validation[['pid', 'passage']]
passa = passa.drop_duplicates()
qid_pid = validation[['qid', 'pid']]

#apply a preprocessing function to the column passage
passa['token_passage'] = passa.apply(lambda x: preprocessing_text(x.passage), axis=1)

#convert the tokenized column into list
tokenized_pass = passa['token_passage'].to_list()

#create a vocabulary
vocabulary = {}

for i in tokenized_pass:
    for j in i:
        if j == " ":
            continue
        if j == "" or i == '':
            continue
        try: 
            vocabulary[j] += 1
        except:
            vocabulary[j] = 1

#removing the stop words from the vocabulary (I think we should remove them given the way our IR works)
stopwords = stopwords.words('english')
#print(stopwords)
for stop_word in stopwords:
    vocabulary.pop(stop_word, None)

#convert the dictionary into list
vocab_list = list(vocabulary.keys())

#create the inverted index dictionary
vocab_dict = {}
for key in vocab_list:
    vocab_dict[key] = []

#creating an inveerted index
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


# BM25 model
#assuming no relevance information
r = 0
R = 0

k_1 = 1.2
k_2 = 100
b = 0.75

#calculate document length for each of the passages
document_length = passa[['pid', 'passage']]
document_length = document_length.drop_duplicates() #drop the duplicates

from nltk.corpus import stopwords
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

#Compute tf-idf for the documents
frames = []
#create dataset by pid and by word
for i in vocab_dict:
    list_id = []
    term_freq = []
    for j in vocab_dict[i]:
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
for i in vocab_dict:
    
    word.append(i)
    n = len(vocab_dict[i])
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
validation = pd.read_csv('./data/part2/validation_data.tsv',sep = '\t')
#subset the relevant columns
queries = validation[['qid', 'queries']]
queries = queries.drop_duplicates()

#let's find term frequency for each of the queries
qid = queries['qid'].tolist()
query_list = queries['queries'].tolist()


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


#Continue BM25

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

#save the file as a csv file
final_bm25.to_csv("bm25.csv", header = False, index = False)

#select the top 100 paragraphs per query
top_100_par = final_bm25.groupby(['qid']).head(100)

#save the file as a csv file
top_100_par.to_csv("bm25_top100.csv", header = False, index = False)

#load the bm25 model back to memory
colnames=['qid', 'pid', 'bm25']
final_bm25 = pd.read_csv(r"bm25.csv", names=colnames)




#Evaluation metrics

#Calculate mAP for k = 3
mAP = calculate_mAP(3, validation, final_bm25)
print("The mean average precision at rank 3 is", mAP)

#Calculate mAP for k = 10
mAP = calculate_mAP(10, validation, final_bm25)
print("The mean average precision at rank 10 is", mAP)

#Calculate mAP for k = 100
mAP = calculate_mAP(100, validation, final_bm25)
print("The mean average precision at rank 100 is", mAP)

#Calculate nDCG

#sort the validation set
validation_sorted = validation.sort_values(by=['qid', 'relevancy'], ascending=False)

#calculate the ndcg score for k = 3
ndcg_k = calculate_ndcg(3, validation, final_bm25, validation_sorted)
avg_ndcg = ndcg_k['ndcg'].sum() / len(ndcg_k)
print("The average nDCG score is", avg_ndcg)

#calculate the ndcg score for k  = 10
ndcg_k = calculate_ndcg(10, validation, final_bm25, validation_sorted)
avg_ndcg = ndcg_k['ndcg'].sum() / len(ndcg_k)
print("The average nDCG score is", avg_ndcg)

#calculate the ndcg score for k = 100
ndcg_k = calculate_ndcg(100, validation, final_bm25, validation_sorted)
avg_ndcg = ndcg_k['ndcg'].sum() / len(ndcg_k)
print("The average nDCG score is", avg_ndcg)














