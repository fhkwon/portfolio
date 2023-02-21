import datetime
import pandas as pd
from collections import namedtuple
from tensorflow.python.keras.initializers import RandomNormal

def to_zero(x):
    try:
        x = str(round(x))
        x = pd.to_datetime(x, format="%Y%m%d")
        x = x.date().strftime("%Y%m%d")[:4]
        current_dt = datetime.date.today().strftime("%Y%m%d")[:4]
        x = int(current_dt)-int(x) #BirthDate => int
        
        if x < 10: #10대 아파트 조식 존재할 수 있기 때문에 -> 10살 아래는 이상치라고 생각해서 다 0으로 변환
            x=0
    except:
        x = 0
    return x

def fill_birth(df):
    mean_ages = dict(df[df['Birth'] != 0].groupby('GroupId').Birth.mean())
    for i in range(len(df)):
        if df.iloc[i, 1] == 0:
            if df.iloc[i,0] in mean_ages:
                df.iloc[i,1] = round(mean_ages[df.iloc[i,0]])
            else:
                df.iloc[i,1] = round(mean_ages[0])
        else:
            pass
    return df

def drop_foodtag(food_df, df):
    food_cols = list(food_df.columns)
    food_cols.remove('FoodId')
    if df[food_cols].isna().sum().sum() > 0:
        df.dropna(subset=food_cols, inplace=True)
    return df

def fill_users(users_df, df):
    user_cols = users_df.columns.to_list()
    user_cols.remove('UserId')

    max_values = {}
    for i in user_cols:
        max_values[i] = max(users_df[i].values)+1
        
    if df[user_cols].isna().sum().sum() > 0:
        print(f'Warning: there are {df[user_cols].isna().sum().sum()} null values in total for User tags. Please fill the values for a better prediction')
        df.fillna(value = max_values, inplace=True)
    return df, max_values

def split_xy(df):    
    sparse_features = df.columns.to_list()
    sparse_features.remove('selected')
    target = ['selected']
    return sparse_features, target

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=3, use_hash=False, vocabulary_path=None, dtype="int32", embeddings_initializer=None,
                embedding_name=None, group_name='default_group', trainable=True):
        if embedding_dim == "auto":
            embedding_dim = 8 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)
        if embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()