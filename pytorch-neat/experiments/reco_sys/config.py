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
from sklearn.metrics import r2_score


def get_inputs_from_csv(train_path):
    '''
    get inputs from the already splittled dataset
    :param train_path:
    :return:
    '''
    ratings = pd.read_csv(
        r"C:\Users\laris\Documents\Github\pytorch-neat\neat\experiments\reco_sys\datasets\ml-latest-small\ratings.csv")  # we take the small dataset #TODO: delete this?

    train_ratings = pd.read_csv(train_path)

    #scale the ratings
    scaler = preprocessing.MinMaxScaler()
    train_ratings['rating'] = scaler.fit_transform(train_ratings[['rating']])

    all_movieIds = train_ratings['movieId'].unique()

    num_users = train_ratings['userId'].max() + 1
    num_items = train_ratings['movieId'].max() + 1

    return scaler, num_users, num_items, train_ratings, all_movieIds


class RecoSysConfig:
    np.random.seed(123)  # for reproducibility

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    scaler, num_users, num_items, train_ratings, all_movieIds = get_inputs_from_csv(r"C:\Users\laris\PycharmProjects\KaggleRS\dataset\test_dataset_used.csv")
    print(f"LEN TRAIN: {len(train_ratings)}")
    #workaround
    num_items = 193610
    num_users = 611
    embed_dim = 50  # current emb dim used

    model = NCF(scaler, embed_dim, num_users, num_items, train_ratings, all_movieIds)
    model.load_state_dict(torch.load(r'C:\Users\laris\PycharmProjects\KaggleRS\saved_weights\weights_only_e25_emb_size50.pth'))

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

    NUM_INPUTS = 100
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'  # TODO: try different activation fc
    SCALE_ACTIVATION = 4.9

    POPULATION_SIZE = 0  # will be set below
    NUMBER_OF_GENERATIONS = 0  # will be set below

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

    @staticmethod
    def set_global_var(index_population):
        sizes_to_try = [20, 50, 100, 150, 200]
        RecoSysConfig.NUMBER_OF_GENERATIONS = 1500
        RecoSysConfig.POPULATION_SIZE = sizes_to_try[index_population]

    def fitness_fn(self, genome):
        fitness = 1*len(self.inputs)
        phenotype = FeedForwardNet(genome, self)

        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(input)
            loss = np.sum(np.abs(pred.data.numpy().astype(float) - target.data.numpy().astype(float)))/len(target) # MAE
            fitness -= loss

        return fitness/len(self.inputs)


    def get_preds_and_labels(self, genome, generation_number):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        csv_path = r"C:\Users\laris\PycharmProjects\KaggleRS\experiments\predictions"
        data = []
        for input, target in zip(self.inputs, self.targets):
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = float(phenotype(input))
            target = float(target)
            data.append([pred, target,
                         self.scaler.inverse_transform(np.full((1, 1), pred))[0][0],
                         self.scaler.inverse_transform(np.full((1, 1), target))[0][0]],)

        pred_df = pd.DataFrame(data=data, columns=["pred_rating_scaled", "real_rating_scaled", "pred_rating",
                                             "real_rating"])
        pred_df.to_csv(f"{csv_path}/NEAT_predictions_gen_{generation_number}_g{self.NUMBER_OF_GENERATIONS}_p{self.POPULATION_SIZE}.csv", index=False)
        # we log the number of generation where it stopped, the total number of generations and the population size

        mae = (abs(pred_df['pred_rating'] - pred_df['real_rating'])).sum() / len(pred_df)
        r2 = r2_score(pred_df['real_rating'], pred_df['pred_rating'])

        return mae, r2




