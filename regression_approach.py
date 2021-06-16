# base : https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems
# saved_models a regression problem of user rating a movie

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import r2_score
import time
import json
from tqdm import tqdm

PARAMS = {'learning_rate': 0.001,
          'optimizer': 'Adam',
          'epochs': 25,
          'embedding_size': [20, 50, 100, 150, 200]}


def split_data(df, number_reviews):
    """
    we'll use the leave-one-out methodology, using the last 3 reviewed movie for each user in the test set
    :param df: dataset to be splitted
    :return:
    """

    ranks_test = list(range(1, number_reviews + 1))
    df['ranked_latest'] = df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test_df = df[df['ranked_latest'].isin(ranks_test)]
    train_df = df[~df['ranked_latest'].isin(ranks_test)]

    train_df = train_df[['userId', 'movieId', 'rating']]
    test_df = test_df[['userId', 'movieId', 'rating']]

    #shuffle the datasets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    print(f"len test df: {len(test_df)}")
    return train_df, test_df


class MovieLensTrainDataset(Dataset):
    """MovieLens Pytorch Dataset for Training
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
    """

    def __init__(self, ratings):
        self.users, self.items, self.labels = self.get_dataset(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings):

        users, items, labels = [], [], []

        user_item_set = set(zip(ratings['userId'], ratings['movieId']))  # set of items that the user interacted with

        for (u, i) in (user_item_set):
            users.append(u)
            items.append(i)
            labels.append(ratings[(ratings['userId'] == u) & (ratings['movieId'] == i)].values[0][2])

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            scaler: scaler of the input
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, scaler, emb_dim, num_users, num_items, ratings, all_movieIds):
        super().__init__()

        #TODO: add a bias to the embeddings?
        #TODO: add dropout?
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.fc1 = nn.Linear(in_features=emb_dim*2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        self.scaler = scaler
        self.save_hyperparameters('num_users', 'num_items', 'ratings', 'all_movieIds')

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        #concat the 2 embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)  #TODO try with dot product also?; average the embeddings?

        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector)) #TODO: Softmax and model the problem as a muticlass one?

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.L1Loss()(predicted_labels, labels.view(-1, 1).float())

        return loss

    def training_epoch_end(self, outputs):
        print(f"train out: {outputs}")

        loss = torch.stack([output['loss'] for output in outputs]).mean()
        neptune_logger.log_metric('train_loss/epoch', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings.iloc[:-1500, :]), #the last 1500 ratings we leave for validation
                          batch_size=64, num_workers=8)

    def val_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings.iloc[-1500:, :]),
                          batch_size=128, num_workers=8) #TODO: bigger batch size


    def validation_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.L1Loss()(predicted_labels, labels.view(-1, 1).float())
        self.log('val_loss', loss)

        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output for output in outputs]).mean()
        neptune_logger.log_metric('val_loss/epoch', loss)


if __name__ == "__main__":
    np.random.seed(123)  # for reproducibility

    with open('credentials_neptune.json') as json_file:
        data = json.load(json_file)


    # we take the small dataset #TODO: try with bigger ones
    ratings = pd.read_csv(
        r"C:\Users\laris\Documents\Github\pytorch-neat\neat\experiments\reco_sys\datasets\ml-latest-small\ratings.csv")
    # print(ratings.head())
    # print(f"Ratings dimensions: {ratings.shape}")


    print(f"There are {len(ratings)} rows of data from {len(set(ratings['userId'].values))} users.")
    # print(ratings.sample(10))  # Each row corresponds to a movie review made by a single user

    #scale the ratings
    scaler = preprocessing.MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    train_ratings, test_ratings = split_data(ratings, 3)

    all_movieIds = ratings['movieId'].unique()

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1

    for i in tqdm(range(5)):
        start = time.time()

        neptune_logger = NeptuneLogger(
            api_key=data['api_key'],
            project_name=data['project_name'],
            params=PARAMS)

        model = NCF(scaler, PARAMS['embedding_size'][i], num_users, num_items, train_ratings, all_movieIds)

        #train the model
        trainer = pl.Trainer(max_epochs=PARAMS['epochs'], reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50,
                             logger=neptune_logger, callbacks=[pl.callbacks.ModelCheckpoint(dirpath=f"./saved_models/embedding{PARAMS['embedding_size'][i]}"),
                                                               EarlyStopping(monitor='val_loss')],
                             check_val_every_n_epoch=1)

        trainer.fit(model)

        # save the model weights
        torch.save(model.state_dict(), f'./saved_weights/weights_only_e{PARAMS["epochs"]}'
                                       f'_emb_size{PARAMS["embedding_size"][i]}.pth')

        # reload model
        # saved_model = NCF.load_from_checkpoint("./saved_models/epoch=9-step=359.ckpt")
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
            # print(f"{scaler.inverse_transform(predicted_labels.reshape(-1, 1))[0][0]} "
            #       f"real rating: "
            #       f"{scaler.inverse_transform(test_ratings[(test_ratings['userId'] == u)  & (test_ratings['movieId']==i)].values[0][2].reshape(-1, 1))[0][0]}")

            # pred_arr.append([u, i, format(scaler.inverse_transform(predicted_labels.reshape(-1, 1))[0][0], '.3f'),
            #                  format(scaler.inverse_transform(test_ratings[(test_ratings['userId'] == u) &
            #                                                               (test_ratings['movieId'] == i)].values[0][2].reshape(-1, 1))[0][0], '.3f')])
            pred_arr.append([u, i, format(predicted_labels.reshape(-1, 1)[0][0], '.3f'),
                             format(test_ratings[(test_ratings['userId'] == u) &
                                                                          (test_ratings['movieId'] == i)].values[0][2], '.3f'),
                             format(scaler.inverse_transform(predicted_labels.reshape(-1, 1))[0][0], '.3f'),
                                              format(scaler.inverse_transform(test_ratings[(test_ratings['userId'] == u) &
                                                                                           (test_ratings['movieId'] == i)].
                                                                              values[0][2].reshape(-1, 1))[0][0], '.3f')
                             ])

        pred_df = pd.DataFrame(data=pred_arr, columns=["userId", "movieId", "pred_rating_scaled", "real_rating_scaled",
                                                       "pred_rating", "real_rating"])
        pred_df.to_csv("predictions.csv", index=False)

        pred_df[["pred_rating", "real_rating"]] = pred_df[["pred_rating", "real_rating"]].astype(float)

        print(f"MAE test set : {(abs(pred_df['pred_rating']-pred_df['real_rating'])).sum()/len(pred_df)}")
        print(f"R^2 test set : {r2_score(pred_df['real_rating'], pred_df['pred_rating'])}")

        print(f"Execution time: {(time.time()-start)/60} mins")
