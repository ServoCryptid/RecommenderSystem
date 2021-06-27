import pandas as pd
from sklearn.model_selection import train_test_split
'''
As a first approach we split the small dataset in 70% train and 30% test (after we shuffle the rows)

Update 19.05.2021: we need a smaller dataset for debugging; Approach: take the most 10 active users
'''


if __name__ == "__main__":
    df = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    # print(df.head(15))

    # shuffle dataframme rows
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df.head(15))

    grp = df.groupby(["userId"])["rating"].count().reset_index()
    grp.rename(columns={'rating': 'rating_count'}, inplace=True)

    important_users = grp.sort_values("rating_count", ascending=False).head(10)['userId'].tolist()
    print(important_users)

    df_imp_users = df[df['userId'].isin(important_users)]
    print(df_imp_users)

    X = df_imp_users[['userId', 'movieId']]
    # print(X)

    y = df_imp_users['rating']
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
    df_train = pd.DataFrame(X_train, columns=['userId', 'movieId'])
    df_train['rating'] = y_train
    print(df_train)
    df_train.to_csv('datasets/ml-latest-small/splitted/10users_train.csv', index=False)

    df_test = pd.DataFrame(X_test, columns=['userId', 'movieId'])
    df_test['rating'] = y_test
    print(df_test)
    df_test.to_csv('datasets/ml-latest-small/splitted/10users_test.csv', index=False)



