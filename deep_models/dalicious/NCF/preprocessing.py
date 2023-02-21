import pandas as pd
import random

def preprocess_data(df):
    for i in list(set(df['UserId'])):
        if len(df.loc[df['UserId']==i]) < 2:
            df.drop(df.loc[df['UserId'] == i].index, inplace=True)
    user2id = {w: i for i, w in enumerate(list(set(df['UserId'])))}
    item2id = {w: i for i, w in enumerate(list(set(df['FoodId'])))}
    df['UserId'] = df['UserId'].apply(lambda x: user2id[x])
    df['FoodId'] = df['FoodId'].apply(lambda x: item2id[x])
    item_pool = set(df['FoodId'].unique())
    num_users = df['UserId'].nunique()+1
    num_items = df['FoodId'].nunique()+1
    return df, user2id, item2id, item_pool, num_users, num_items


def train_test_split(df, item_pool, num_ng_test, num_neg):
    df = df.sort_values(by=['UserId', 'orderdate'])
    df['rank_latest'] = df.groupby(['UserId'])['orderdate'].rank(method='first', ascending=False)
    test = df.loc[df['rank_latest'] == 1]
    train = df.loc[df['rank_latest'] > 1]
    train_ratings = train.drop(columns=['rank_latest', 'orderdate'])
    test_ratings = test.drop(columns=['rank_latest', 'orderdate'])
    interact_status = (df.groupby('UserId')['FoodId'].apply(set).reset_index().rename(columns={'FoodId': 'interacted_items'}))
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), num_ng_test))
    negatives = interact_status[['UserId', 'negative_items', 'negative_samples']]
    train_ratings = pd.merge(train_ratings, negatives[['UserId', 'negative_items']], on='UserId')
    train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_neg))
    test_ratings = pd.merge(test_ratings, negatives[['UserId', 'negative_samples']], on='UserId')
    return train_ratings, test_ratings


def make_train_lists(train_ratings, num_neg):
    user_input, item_input, labels = [], [], []
    for row in train_ratings.itertuples():
        user_input.append(int(row.UserId))
        item_input.append(int(row.FoodId))
        labels.append(int(row.label))
        for i in range(num_neg):
            user_input.append(int(row.UserId))
            item_input.append(int(row.negatives[i]))
            labels.append(int(0))  
    return user_input, item_input, labels

def make_test_lists(test_ratings):
    ts_ui, neg_items =[], []
    for row in test_ratings.itertuples():
        ts_ui.append([int(row.UserId),int(row.FoodId)])
        neg_items.append(row.negative_samples)
    return ts_ui, neg_items