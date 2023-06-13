#import relevant libraries
import pandas as pd
from sklearn.utils import resample
import re
from gensim.models import KeyedVectors
import gensim
from gensim.models import Word2Vec
import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api

#Functions

#Perfrom balanced sampling
def balanced_sampling(df, n_samples):
    """
    Perform balanced sampling on a dataframe with qid, pid, and relevancy columns.
    Returns a new dataframe with an equal number of positive and negative samples for each query.

    Args:
    df (pandas.DataFrame): The input dataframe.
    n_samples (int): The number of samples to take for each query.

    Returns:
    pandas.DataFrame: A new dataframe with balanced samples.
    """
    # Get the unique query ids
    query_ids = df['qid'].unique()

    # Create empty lists to store the positive and negative samples
    pos_samples = []
    neg_samples = []

    # Loop through each query
    for qid in query_ids:
        # Get the positive and negative samples for the current query
        query_df = df[df['qid'] == qid]
        pos_df = query_df[query_df['relevancy'] == 1]
        neg_df = query_df[query_df['relevancy'] == 0]

        # Sample n_samples from each group
        pos_sample = pos_df.sample(n=min(n_samples, len(pos_df)), replace=True)
        neg_sample = neg_df.sample(n=min(n_samples*5, len(neg_df)), replace=True)

        # Add the samples to the lists
        pos_samples.append(pos_sample)
        neg_samples.append(neg_sample)

    # Concatenate the samples into a new dataframe
    pos_df = pd.concat(pos_samples)
    neg_df = pd.concat(neg_samples)
    new_df = pd.concat([pos_df, neg_df])

    # Shuffle the new dataframe
    new_df = new_df.sample(frac=1).reset_index(drop=True)

    return new_df


def subset_and_tokenize(column_1, column_2, train_data):
    '''
    A function that subsets columns and get the unique values per column
    and tokenizes the values in the column which indludes free text
    
    Inputs: column_1 - the first column we want to keep (str)
            column_2 - the second column we want to keep (str)
            train_data - the dataset which we are subsetting (dataframe)
    Otuputs: a dataframe with the subsetted columns and the free text tokenized (dataframe)
    '''
    
    #subset the unique values in dataframe
    unique_d = train_data[[column_1, column_2]]
    unique_d = unique_d.drop_duplicates()
    #tokenize the free text column
    unique_d['tokenized'] = unique_d.apply(lambda x: preprocessing_text(x[column_2]), axis=1)
    
    return unique_d


def preprocessing_text(text):
    '''
    Input: a whole passage as a string (type: str)
    Pre-process the text to remove puncuations and turn to lower case
    After that convert the text into a list of tokens (tokenization)
    Return: list of tokens
    '''

    text = str(text)
    text = text.replace('\n', " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    tokens = text.split(" ") #split based on spaces
    tokens = list(filter(None, tokens))
    
    return tokens


def calculate_avg_embedding(list_words, model):
    '''
    Calculate the average word embeddings for a given list of words
    
    Inputs: List_words - a list of words for which to calculate the word embeddings (list)
            model - the pre-trained word2vec model
            
    Outputs: the average word emvedding for the list of words (np.array of size (300))
    '''
    len_l = len(list_words)
    empty_arr = np.empty([len_l,300])
    
    for i in range(len_l):
        
        try:
            empty_arr[i] = model[list_words[i]] 
        except:
            empty_arr[i] = np.zeros(300)
            
    
    mean_embedding = np.mean(empty_arr, axis = 0)
    
    return mean_embedding


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
    top_par.loc[:,'rank'] = top_par.groupby('qid')['predictions'].rank(method='dense', ascending=False)
    
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


class LSTMRanker(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMRanker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = self.fc(out[:, -1, :])
        return out





#load the train and test datasets
train_data = pd.read_csv(r'./data/part2/train_data.tsv', sep = '\t')
test_data = pd.read_csv(r'./data/part2/validation_data.tsv', sep = '\t')


#sample the trained data
train_data_sample = balanced_sampling(train_data, 5)

#tokenize the unique queries for the train and test data
unique_queries = subset_and_tokenize('qid', 'queries', train_data_sample)
unique_queries_v = subset_and_tokenize('qid', 'queries', test_data)

#tokenize the passages
unique_passages = subset_and_tokenize('pid', 'passage', train_data_sample)

#Word Embeddings
#use Google's pretrained word2vec model
#model = gensim.models.KeyedVectors.load_word2vec_format('word2vec-google-news-300.gz', binary=True)
path = api.load("word2vec-google-news-300", return_path=True)
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

#calculate the average query and passage embeddings
unique_queries['avg_embedding'] = unique_queries.apply(lambda x: calculate_avg_embedding(x.tokenized, model), axis=1)
unique_passages['avg_embedding'] = unique_passages.apply(lambda x: calculate_avg_embedding(x.tokenized, model), axis=1)

#merge the passage and query embeddings backt to the train dataset
merged_train_data = pd.merge(train_data_sample, unique_queries, on= 'qid', how = 'left')
merged_train_data = pd.merge(merged_train_data, unique_passages, on = 'pid', how = 'left')

#keep only the relevant columns and rename them
merged_train_data = merged_train_data[['qid', 'pid', 'queries_x', 'passage_x', 'relevancy', 'tokenized_x', 'avg_embedding_x', 'tokenized_y', 'avg_embedding_y']]
merged_train_data = merged_train_data.rename(columns={"queries_x": "queries", "passage_x": "passage", "tokenized_x":"tokenized_q", "avg_embedding_x":"avg_embedding_q", "tokenized_y": "tokenized_p", "avg_embedding_y":"avg_embedding_d"})

#calculate the average embeddings for the queries and the passages
unique_queries_v['avg_embedding'] = unique_queries_v.apply(lambda x: calculate_avg_embedding(x.tokenized, model), axis=1)
#load the test passages along with their embeddings
unique_passages_v = pd.read_feather('passages_v')

#merge the passage and query embeddings backt to the train dataset
merged_test_data = pd.merge(test_data, unique_queries_v, on= 'qid', how = 'left')
merged_test_data = pd.merge(merged_test_data, unique_passages_v, on = 'pid', how = 'left')

#keep only the relevant columns and rename them
merged_test_data = merged_test_data[['qid', 'pid', 'queries_x', 'passage_x', 'relevancy', 'tokenized_x', 'avg_embedding_x', 'tokenized_y', 'avg_embedding_y']]
merged_test_data = merged_test_data.rename(columns={"queries_x": "queries", "passage_x": "passage", "tokenized_x":"tokenized_q", "avg_embedding_x":"avg_embedding_q", "tokenized_y": "tokenized_p", "avg_embedding_y":"avg_embedding_d"})

#Concatenate the embeddings
merged_train_data['concatenated_qd'] = merged_train_data.apply(lambda x: np.concatenate([x.avg_embedding_q, x.avg_embedding_d]), axis = 1)
merged_test_data['concatenated_qd'] = merged_test_data.apply(lambda x: np.concatenate([x.avg_embedding_q, x.avg_embedding_d]), axis = 1)

##LSTM implementation
#connect to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

merged_train_data['concatenated_qd_f'] = merged_train_data.apply(lambda x: x['concatenated_qd'].astype('float32'), axis=1)

#Get the X_train and y_train
X_train = concatenated_array = np.stack(merged_train_data['concatenated_qd_f'].values, axis=0)
y_train = merged_train_data['relevancy'].to_numpy()

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

#create a dataloader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the LSTM model
input_size = 600 # input size
hidden_size = 128 # hidden state size
output_size = 1 # output size
model = LSTMRanker(input_size, hidden_size, output_size)
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the LSTM model
num_epochs = 100
for epoch in range(num_epochs):

    print("Here")
    for i, (inputs, targets) in enumerate(dataloader):
        
        
        # Forward pass
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs.squeeze(), targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring
        if (epoch+1) % 10 == 0 and (i+1) % len(dataloader) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))


# Save the trained model
torch.save(model.state_dict(), 'lstm_model.pt')
print("Model trained")

#Evaluate the model on the validation dataset
#Load the model 
model.load_state_dict(torch.load('lstm_model.pt', map_location=torch.device('cpu')))
merged_test_data['concatenated_qd_f'] = merged_test_data.apply(lambda x: x['concatenated_qd'].astype('float32'), axis=1)

#Get the X_test
X_test = concatenated_array = np.stack(merged_test_data['concatenated_qd_f'].values, axis=0)
X_test = torch.from_numpy(X_test).float()

#Make a prediction
pred = model(X_test)

merged_test_data['predictions'] = pred.detach().numpy()
sub_merged_test_data = merged_test_data[['qid', 'pid', 'predictions']]

#sort values in decending order
sub_merged_test_data = sub_merged_test_data.sort_values(by=['qid', 'predictions'], ascending=False)

mAP = calculate_mAP(100, test_data, sub_merged_test_data)
print("The mean average percision is", mAP)

#sort the test set
validation_sorted = test_data.sort_values(by=['qid', 'relevancy'], ascending=False)

ndcg_k = calculate_ndcg(100, test_data, sub_merged_test_data, validation_sorted)
avg_ndcg = ndcg_k['ndcg'].sum() / len(ndcg_k)
print("The average nDCG score is", avg_ndcg)#



## Run the model on the candidate 1000 passages
#use Google's pretrained word2vec model
#model_1 = gensim.models.KeyedVectors.load_word2vec_format('word2vec-google-news-300.gz', binary=True)
path = api.load("word2vec-google-news-300", return_path=True)
model_1 = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

col = ['qid', 'pid', 'query', 'passage']
candidate_passage = pd.read_csv(r'./data/part2/candidate_passages_top1000.tsv', sep = '\t', names = col)

#tokenize the queries and the passages
#find the unique passages and queries and tokenize them
unique_passages_t = subset_and_tokenize('pid', 'passage', candidate_passage)
unique_queries_t = subset_and_tokenize('qid', 'query', candidate_passage)

#calculate average embeddings for the queries and the passages
unique_passages_t['avg_embedding'] = unique_passages_t.apply(lambda x: calculate_avg_embedding(x.tokenized, model_1), axis=1)
unique_queries_t['avg_embedding'] = unique_queries_t.apply(lambda x: calculate_avg_embedding(x.tokenized, model_1), axis=1)

#merge the passage and query embeddings backt to the train dataset
merged_test_data = pd.merge(candidate_passage, unique_queries_t, on= 'qid', how = 'left')
merged_test_data = pd.merge(merged_test_data,unique_passages_t, on = 'pid', how = 'left')

#keep only the relevant columns and rename them
merged_test_data = merged_test_data[['qid', 'pid', 'query_x', 'passage_x', 'tokenized_x', 'avg_embedding_x', 'tokenized_y', 'avg_embedding_y']]
merged_test_data = merged_test_data.rename(columns={"query_x": "queries", "passage_x": "passage", "tokenized_x":"tokenized_q", "avg_embedding_x":"avg_embedding_q", "tokenized_y": "tokenized_p", "avg_embedding_y":"avg_embedding_d"})

#concatenate the embeddings
merged_test_data['concatenated_qd'] = merged_test_data.apply(lambda x: np.concatenate([x.avg_embedding_q, x.avg_embedding_d]), axis = 1)
merged_test_data['concatenated_qd_f'] = merged_test_data.apply(lambda x: x['concatenated_qd'].astype('float32'), axis=1)

#Get the X_test
X_test = concatenated_array = np.stack(merged_test_data['concatenated_qd_f'].values, axis=0)
X_test = torch.from_numpy(X_test).float()

#make a prediction
pred = model(X_test)

merged_test_data['predictions'] = pred.detach().numpy()
merged_test_data = merged_test_data.sort_values(by=['qid', 'predictions'], ascending=False)

#load the test data in order to order them the same way
colnames = ['qid', 'query']
test_queries = pd.read_csv(r'./data/part2/test-queries.tsv', sep = '\t', names = colnames)

test_queries = test_queries[['qid']]

#left join the data
final_ranking = pd.merge(test_queries, merged_test_data, on = 'qid', how = 'left')

final_ranking['assignment'] = 'A2'
final_ranking['rank'] = final_ranking.groupby("qid")["predictions"].rank(method="dense", ascending=False)
final_ranking['algoname'] = 'NN'

#get the top 100 passages
final_ranking_100 = final_ranking.groupby(['qid']).head(100)
# subset the dataset
final_ranking_100 = final_ranking_100[['qid', 'assignment', 'pid', 'rank', 'predictions', 'algoname']]

# save the dataframe as a text file
final_ranking_100.to_csv('NN.txt', header=False, sep='\t', index=False)












