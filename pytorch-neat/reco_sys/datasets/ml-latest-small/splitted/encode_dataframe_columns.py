import pandas as pd
import pickle


def get_encoding_dictionary(values_set):

    return {value: count for count, value in enumerate(values_set)}


def encode_columns(df, encoding_dict_user, encoding_dict_item, file_name):
    #encode
    df['userId'] = df['userId'].apply(lambda x: encoding_dict_user[x])
    df['movieId'] = df['movieId'].apply(lambda x: encoding_dict_item[x])
    df.head(20)

    #save df
    df.to_csv(r'C:\Users\laris\PycharmProjects\AutoRec\nextit_datasets\{}'.format(file_name))


if __name__ == "__main__":
    df_train_full = pd.read_csv(r"/neat/experiments/reco_sys/datasets/ml-latest-small/splitted/10users_train.csv")
    df_test_full = pd.read_csv(r"/neat/experiments/reco_sys/datasets/ml-latest-small/splitted/10users_test.csv")

    encoded_userId_dict = get_encoding_dictionary(set(df_train_full['userId']).union(df_test_full['userId']))
    encoded_itemId_dict = get_encoding_dictionary(set(df_train_full['movieId']).union(df_test_full['movieId']))

    print(len(encoded_userId_dict))
    print(len(encoded_itemId_dict))
    # df_train_full['encoded_userId'] = df_train_full['userId'].apply(lambda x: encoded_userId_dict[x])
    # print(df_train_full.head(-20))
    #
    # df_train_full['encoded_movieId'] = df_train_full['movieId'].apply(lambda x: encoded_movieId_dict[x])
    # print(df_train_full.head(-20))
    # print(encoded_movieId_dict[53125])  # just for sanity check

    #pickle the dictionaries for decoding in predictions.csv
    with open(r'/pickles/userId_encoding_dict_10users_movielens.pickle', 'wb') as handle:
        pickle.dump(encoded_userId_dict, handle)

    with open(r'/pickles/movieId_encoding_dict_10users_movielens.pickle', 'wb') as handle:
        pickle.dump(encoded_itemId_dict, handle)


    #for reading it
    # with open(r'C:\Users\laris\PycharmProjects\AutoRec\pickles\userId_encoding_dict.pickle', 'rb') as handle:
    #     read_dict = pickle.load(handle)
    #
    # with open(r'C:\Users\laris\PycharmProjects\AutoRec\pickles\movieId_encoding_dict.pickle', 'rb') as handle:
    #     read_dict2 = pickle.load(handle)

    # print(read_dict == encoded_userId_dict)
    # print(read_dict2 == encoded_movieId_dict)

    encode_columns(df_train_full, encoded_userId_dict, encoded_itemId_dict, "10users_movielens_Train_encoded.csv")
    encode_columns(df_test_full, encoded_userId_dict, encoded_itemId_dict, "10users_movielens_Test_encoded.csv")


    #TODO: Reverse / invert a dictionary mapping when decoding predictions.csv - update 16.06: not used anymore


