# from: https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems
# saved_models a classification problem of user interacting/or not with a movie


import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import time


def split_data(df):
    """
    we'll use the leave-one-out methodology, using the last reviewd movie for each user in the test set
    :param df: dataset to be splitted
    :return:
    """

    df['ranked_latest'] = df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    train_df = df[df['ranked_latest'] != 1]
    test_df = df[df['ranked_latest'] == 1]

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

        num_negatives = 4  #TODO: gridsearch the best value for this
        cnt=0
        for (u, i) in (user_item_set):
            print(f"CNT: {cnt}/{len(user_item_set)}")
            cnt+=1
            users.append(u)
            items.append(i)
            labels.append(1)

            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)

                # check that the user has not interacted with this item
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

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

        emb_dim = 8  #TODO: find the best value for it
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        #concat the 2 embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=4)


if __name__ == "__main__":
    np.random.seed(123)  # for reproducibility

    ratings = pd.read_csv("./dataset/movielens_20M/rating.csv", parse_dates=['timestamp'])

    print(ratings.head())
    print(f"Ratings dimensions: {ratings.shape}")
    print(ratings.info())

    # we select only 30% of the users from the dataset => ~ 41.5k users
    # percentage_users = 0.3
    percentage_users = 1 # we take all of them
    rand_userIds = np.random.choice(ratings['userId'].unique(),
                                    size=int(len(ratings['userId'].unique())*percentage_users))
    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    print(f"There are {len(ratings)} rows of data from {len(rand_userIds)} users.")
    print(ratings.sample(10))  # Each row corresponds to a movie review made by a single user

    train_ratings, test_ratings = split_data(ratings)

    train_ratings.loc[:, 'rating'] = 1  # now we have only positive interactions, we need to add "negative" ones too
                                        # we sample from the unrated movies 4 * number ratings given by a user (for 1
                                        # rated movie we have 4 unrated movies)

    print(train_ratings.sample(5))
    all_movieIds = ratings['movieId'].unique()

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1

    model = NCF(num_users, num_items, train_ratings, all_movieIds)

    #train the model
    trainer = pl.Trainer(max_epochs=5, reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50, logger=False,
                         checkpoint_callback=False)

    trainer.fit(model)

    #test the model
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    hits = []

    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(model(torch.tensor([u]*100),
                                            torch.tensor(test_items)).detach().numpy())

        print(f"predicted labels: {predicted_labels}")
        print(f"shape predicted labels: {predicted_labels.shape}")

        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)

    print(f"Hit ratio is: {np.average(hits)}")
