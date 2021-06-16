import sys
sys.path.append(r"C:\Users\laris\PycharmProjects")

import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import time
from sklearn import preprocessing
from KaggleRS.regression_approach import NCF, split_data

global A


def compute_a(ratings_list):
    '''
    A as the maximum error of a recommendation system
    in which its outputs fall within the range of valid rating values
    :return: A
    '''

    sum = 0
    for rating in ratings_list:
        if rating > 0:
            sum += max(float(rating), (1-float(rating)))  # we use 1 - instead of 5 - because the ratings are scaled
                                                             # to be between (0, 1)

    print(f"Value for A: {sum}")

    return sum


def process_input(): # CURRENTLY NOT USED; ONLY FOR INSPO

    '''
    transform the dataset in a specific way
    #1 inspo: https://arxiv.org/pdf/1708.05031.pdf
    :return:
    '''

    FEATURE_COLUMNS = ['userId', 'movieId']
    # ohe = OneHotEncoder(handle_unknown='ignore')
    mlb = MultiLabelBinarizer(sparse_output=True)

    train_df = pd.read_csv('experiments/reco_sys/datasets/ml-latest-small/splitted/100k_train.csv')
    test_df = pd.read_csv('experiments/reco_sys/datasets/ml-latest-small/splitted/100k_test.csv')

    train_ratings = train_df['rating'].values
    test_ratings = test_df['rating'].values

    print(train_df.head())
    user_set = set(train_df['userId'].append(test_df['userId']))
    print(len(user_set))
    item_set = set(train_df['movieId'].append(test_df['movieId']))
    print(len(item_set))

    train_df = train_df.groupby(['userId'])['movieId'].apply(list).reset_index(name='movieId')
    train_df = train_df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(train_df.pop('movieId')),
            index=train_df.index,
            columns=mlb.classes_))
    print(train_df)

    test_df = test_df.groupby(['userId'])['movieId'].apply(list).reset_index(name='movieId')
    test_df = test_df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(test_df.pop('movieId')),
            index=test_df.index,
            columns=mlb.classes_
        ))
    print(test_df)


    # X_train = ohe.fit_transform(train_df[FEATURE_COLUMNS])
    # X_test = ohe.transform(test_df[FEATURE_COLUMNS])

    return train_df, test_df, train_ratings, test_ratings, user_set, item_set


def get_inputs():
    ratings = pd.read_csv(
        r"C:\Users\laris\Documents\Github\pytorch-neat\neat\experiments\reco_sys\datasets\ml-latest-small\ratings.csv")  # we take the small dataset
    # print(ratings.head())
    # print(f"Ratings dimensions: {ratings.shape}")

    number_ratings = 100
    ratings = ratings.sample(number_ratings)

    #scale the ratings
    scaler = preprocessing.MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    train_ratings, test_ratings = split_data(ratings, 1)

    all_movieIds = ratings['movieId'].unique()

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1

    return scaler, num_users, num_items, train_ratings, all_movieIds


class RecoSysConfig:
    global A

    np.random.seed(123)  # for reproducibility

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    scaler, num_users, num_items, train_ratings, all_movieIds = get_inputs()

    #workaround
    num_items = 193610
    num_users = 611

    model = NCF(scaler, num_users, num_items, train_ratings, all_movieIds)
    model.load_state_dict(torch.load(r'C:\Users\laris\PycharmProjects\KaggleRS\saved_weights\weights_only_e2.pth'))

    # reload pretrained embedding weights
    for name, param in model.named_parameters():
        if name == 'user_embedding.weight':
            # print(f"name: {name} --- weights:{param.data}")
            weight_user_emb = param.data
            embedding_user = nn.Embedding.from_pretrained(weight_user_emb)
            # print(f"EMBEDDING SHAPE: {embedding_user}")

        elif name == 'item_embedding.weight':
            # print(f"name: {name} --- weights:{param.data}")
            weight_item_emb = param.data
            embedding_item = nn.Embedding.from_pretrained(weight_item_emb)

    NUM_INPUTS = 200
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid' #TODO: try different activation fc
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 1  #TODO: - a solution is defined as having a fitness <= this fitness threshold; is 1 for scaled ratings, 5 otherwise


    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    inputs = []
    for index, row in train_ratings[['userId', 'movieId']].iterrows():
        inputs.append(torch.cat((embedding_user(torch.LongTensor([row['userId']])),
                                 embedding_item(torch.LongTensor([row['movieId']]))), 1))

    targets = list(map(lambda x: autograd.Variable(torch.Tensor([x])), train_ratings['rating']))

    A = compute_a(train_ratings['rating'])

    def fitness_fn(self, genome):
        fitness = A
        phenotype = FeedForwardNet(genome, self)

        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(input)
            # print(f"PRED: {pred.data.numpy().astype(float)}")
            # print(f"TARGETS: {target.data.numpy().astype(float)}")
            loss = np.sum(np.abs(pred.data.numpy().astype(float) - target.data.numpy().astype(float)))/len(target) # MAE
            # print(f"LOSS: {loss}")
            fitness -= loss

        return fitness

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels



