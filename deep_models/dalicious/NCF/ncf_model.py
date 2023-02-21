import numpy as np

from keras.layers import Embedding, Input, Dense, Multiply, concatenate, Flatten
from keras.models import Model
from keras.layers.core import Dense
from keras.regularizers import l2

def NCF(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    # embedding 전 input 레이어를 만들자.
    user_input_layer = Input(shape=(1,), dtype= 'int32', name = 'user_input_layer')
    item_input_layer = Input(shape=(1,), dtype='int32', name = 'item_input_layer')
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                    embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                    embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                    embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                    embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    mf_user_latent = Flatten()(MF_Embedding_User(user_input_layer))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input_layer))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) 

    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input_layer))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input_layer))
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])
    
    '''lecun = 유니폼이나 normal 분산에서 추출한 랜덤값으로 가중치를 초기화 시키되, 이 확률 분포를 fan in 값으로 조절하자는 아이디어이다. 
    즉, input의 크기가 커질수록 초기화 값의 분산을 작게 만들자는 것이다. 
    ReLU가 나오기 전에 사용한 방법으로 가우시안 분포에서 분산을 X의 원래 분산 정도로 보정한다.
    '''
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)

    model = Model(inputs=[user_input_layer, item_input_layer], 
                    outputs=prediction)
    
    return model


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.hr_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, hr, model):
        score = hr
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(hr, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(hr, model)
            self.counter = 0

    def save_checkpoint(self, hr, model):
        if self.verbose:
            print(f'HR increased ({self.hr_min:.6f} --> {hr:.6f}).')
        self.hr_min = hr