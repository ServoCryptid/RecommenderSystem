# base : https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems
# saved_models a regression problem of user rating a movie

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn import preprocessing
import time


def split_data(df):
    """
    we'll use the leave-one-out methodology, using the last 3 reviewd movie for each user in the test set
    :param df: dataset to be splitted
    :return:
    """

    ranks_test = list(range(1, 4))
    df['ranked_latest'] = df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    train_df = df[df['ranked_latest'].isin(ranks_test)]
    test_df = df[~df['ranked_latest'].isin(ranks_test)]

    train_df = train_df[['userId', 'movieId', 'rating']]
    test_df = test_df[['userId', 'movieId', 'rating']]

    return train_df, test_df


class MovieLensTrainDataset(Dataset):
    """MovieLens Pytorch Dataset for Training
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):

        users, items, labels = [], [], []

        user_item_set = set(zip(ratings['userId'], ratings['movieId']))  # set of items that the user interacted with

        num_negatives = 10  #TODO: gridsearch the best value for this; tried also with 4

        cnt = 0

        for (u, i) in (user_item_set):
            print(f"CNT: {cnt}/{len(user_item_set)}")
            cnt += 1
            users.append(u)
            items.append(i)
            labels.append(ratings[(ratings['userId'] == u) & (ratings['movieId'] == i)].values[0][2])

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()

        emb_dim = 100  #TODO: find the best value for it
        #TODO: add a bias to the embeddings?
        #TODO: add dropout?
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.fc1 = nn.Linear(in_features=200, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        self.save_hyperparameters('num_users', 'num_items', 'ratings', 'all_movieIds')

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        #concat the 2 embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1) # try with dot product also?

        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.L1Loss()(predicted_labels, labels.view(-1, 1).float())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=32, num_workers=8)


if __name__ == "__main__":
    np.random.seed(123)  # for reproducibility
    start = time.time()

    ratings = pd.read_csv(
        r"C:\Users\laris\Documents\Github\pytorch-neat\neat\experiments\reco_sys\datasets\ml-latest-small\ratings.csv")  # we take the small dataset
    print(ratings.head())
    print(f"Ratings dimensions: {ratings.shape}")

    percentage_users = 1

    rand_userIds = np.random.choice(ratings['userId'].unique(),
                                    size=int(len(ratings['userId'].unique())*percentage_users))
    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    print(f"There are {len(ratings)} rows of data from {len(rand_userIds)} users.")
    print(ratings.sample(10))  # Each row corresponds to a movie review made by a single user

    #scale the ratings
    scaler = preprocessing.MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    train_ratings, test_ratings = split_data(ratings)

    all_movieIds = ratings['movieId'].unique()

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1

    model = NCF(num_users, num_items, train_ratings, all_movieIds)

    #train the model
    trainer = pl.Trainer(max_epochs=10, reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50, logger=False,
                         callbacks=pl.callbacks.ModelCheckpoint(dirpath="./saved_models/"))  # took ~30 mins/epoch

    print("----training starting----")
    trainer.fit(model)
    print("----end training----")

    # reload model
    # saved_model = NCF.load_from_checkpoint("./saved_models/epoch=19-step=28399_10_neg_samples.ckpt")
    # saved_model.eval()

    #test the model
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    pred_arr = []
    for (u, i) in test_user_item_set:
        predicted_labels = np.squeeze(model(torch.tensor([u]),
                                            torch.tensor([i])).detach().numpy())
        # predicted_labels = np.squeeze(saved_model(torch.tensor([u]),
        #                                           torch.tensor([i])).detach().numpy())

        print(f"{scaler.inverse_transform(predicted_labels.reshape(-1, 1))[0][0]} "
              f"real rating: "
              f"{scaler.inverse_transform(test_ratings[(test_ratings['userId'] == u)  & (test_ratings['movieId']==i)].values[0][2].reshape(-1, 1))[0][0]}")

        pred_arr.append([u, i, format(scaler.inverse_transform(predicted_labels.reshape(-1, 1))[0][0], '.3f'),
                         format(scaler.inverse_transform(test_ratings[(test_ratings['userId'] == u) &
                                                                      (test_ratings['movieId'] == i)].values[0][2].reshape(-1, 1))[0][0], '.3f')])

    pred_df = pd.DataFrame(data=pred_arr, columns=["userId", "movieId", "pred_rating", "real_rating"])
    pred_df.to_csv("predictions.csv", index=False)
    print(f"MAE test set : {(abs(pred_df['pred_rating']-pred_df['real_rating'])).sum()/len(pred_df)}")

    print(f"Execution time: {(time.time()-start)/60} mins")
