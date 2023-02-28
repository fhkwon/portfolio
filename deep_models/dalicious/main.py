import pandas as pd
import numpy as np
import argparse

from NCF.ncf_train import train_ncf
from NCF.ncf_predict import predict_ncf
from DeepFM.deepfm_train import train_deepfm
from DeepFM.deepfm_predict import predict_deepfm

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Hybrid Model.")
    parser.add_argument('--train', nargs='?', default=True,
                        help='Willing to train data?.')
    parser.add_argument('--db_path', nargs='?', default='../../data/',
                        help='Input data path.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--embedding_size', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP의 각 레이어들의 사이즈. # 첫번째 레이어는 두개의 임베딩 결과를 concat 한것이기때문에 layers[0]/2가 embedding size가 됩니다.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='MF_embedding에서 값이 너무 커지는 것을 방지하기 위해 (overfitting 방지) 설정해주는 수.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="MLP의 각 레이어들의 정규화 수. 첫번째 수(reg_layers[0])가 Embedding 단계에서의 정규화 수. 위와 마찬가지로 overfitting 방지.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='train set에 사용할 neg sample의 수. point-wise learning이기 때문에 neg-sample을 따로 뽑아서 모아줘야 한다.')
    parser.add_argument('--num_ng_test', type=int, default=99,
                        help='test set에 사용할 negative test item 수를 정하자')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--ncf_learner', nargs='?', default='adam',
                        help='Specify an optimizer for ncf model: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--deepfm_learner', nargs='?', default='adamax',
                        help='Specify an optimizer for deepfm model: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='각 iter 마다 출력할 길이. 보통 0 은 출력하지 않고, 1은 자세히, 2는 함축적인 정보만 출력하는 형태.')
    parser.add_argument('--out', type=int, default=1,
                        help='트레인한 모델을 저장할지 말지. >0이면 저장. ')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help=" MF에 대해 학습된 모델을 쓸지말지. ''이면 쓰지 않음. ")
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help="MLP에 대해 학습된 모델을 쓸지말지. ''이면 쓰지 않음. ")
    parser.add_argument('--topK', nargs='?', default=10,
                        help="evaluation할 때 상위 몇개의 결과에 대해서 할 건지 정해주세요. ")
    parser.add_argument('--order_path', nargs='?', default='raw_data/order/order_raw.csv',
                        help="주문 데이터를 불러올 경로를 설정해 주세요.")
    parser.add_argument('--user_group_path', nargs='?', default='pps_data/particular_able/able_more_user.csv',
                        help="현재 주문 가능한 유저와 그룹 데이터를 불러올 경로를 설정해주세요.")
    parser.add_argument('--food_makers_path', nargs='?', default='pps_data/able/able_food_makers.csv',
                        help="현재 주문 가능한 음식과 메이커스 데이터를 불러올 경로를 설정해주세요. ")
    parser.add_argument('--save_path', nargs='?', default='outputs',
                        help="결과물 저장시 사용할 파일 경로를 설정해주세요. ")
    parser.add_argument('--model_load_file', nargs='?', default='Pretrain/NeuMF_2023-02-16_11_18_02.h5',
                        help="저장해놓은 모델 가중치를 불러올 경로를 설정해주세요.")
    parser.add_argument('--dnn_hidden_units', nargs='?', default=(400, 400, 400),
                        help="DNN 레이어 구조를 설정해주세요. ")
    parser.add_argument('--l2_reg_dnn', nargs='?', default=0,
                        help="DeepFM 초기화 설정을 알려주세요.")
    parser.add_argument('--seed', nargs='?', default=1024,
                        help="seed 점수를 알려주세요.")
    parser.add_argument('--dnn_dropout', nargs='?', default=0.5,
                        help="dropout 비율을 알려주세요. ")
    parser.add_argument('--dnn_activation', nargs='?', default='relu',
                        help="DNN activation 함수를 설정해주세요.")
    parser.add_argument('--dnn_use_bn', nargs='?', default=False,
                        help="DNN Batch normalization을 하시겠습니까?")
    parser.add_argument('--validation_split', nargs='?', default=0.2,
                        help="validation split을 설정해주세요")
    parser.add_argument('--patience', nargs='?', default=10,
                        help="earlystoppping patience 설정해주세요")
    parser.add_argument('--cls_order_path', nargs='?', default='raw_data/order/cls_data.csv',
                        help="클릭 데이터 주문 데이터를 불러올 경로를 설정해 주세요.")
    parser.add_argument('--cls_food_path', nargs='?', default='raw_data/food/foodtag_raw.csv',
                        help="클릭 음식 데이터를 불러올 경로를 설정해 주세요.")
    parser.add_argument('--cls_user_path', nargs='?', default='raw_data/user/user_preference_notapt.csv',
                        help="클릭 유저 데이터를 불러올 경로를 설정해 주세요.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train = args.train
    db_path = args.db_path
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_neg = args.num_neg
    num_ng_test = args.num_ng_test
    lr = args.lr
    ncf_learner = args.ncf_learner
    deepfm_learner = args.deepfm_learner
    verbose = args.verbose
    out = args.out
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
    topK = args.topK
    order_path = args.order_path
    user_group_path = args.user_group_path
    food_makers_path = args.food_makers_path
    save_path = args.save_path
    model_load_file = args.model_load_file
    DEFAULT_GROUP_NAME = "default_group"
    fm_group = (DEFAULT_GROUP_NAME,)
    dnn_hidden_units=args.dnn_hidden_units
    l2_reg_dnn=args.l2_reg_dnn
    seed=args.seed
    dnn_dropout=args.dnn_dropout
    dnn_activation=args.dnn_activation
    dnn_use_bn=args.dnn_use_bn
    validation_split=args.validation_split
    patience=args.patience
    cls_order_path=args.cls_order_path
    cls_food_path=args.cls_food_path
    cls_user_path=args.cls_user_path

    num_users, num_items, model_out_file, user2id, item2id = train_ncf(db_path, order_path, num_ng_test, num_neg, embedding_size, 
                                                                       layers, reg_layers, reg_mf, ncf_learner, lr, num_epochs, batch_size, verbose, out, topK)
    
    ncf_user_food_df, ncf_group_makers_df, ncf_user_makers_df = predict_ncf(num_users, num_items, embedding_size, layers, reg_layers, reg_mf, 
                                                                            model_out_file, ncf_learner, lr, db_path, user_group_path, food_makers_path, user2id, item2id, save_path='NCF/outputs', model_name='ncf')
    
    model_out_file, user2id, item2id, users_df, food_df, feature_named_tuple, sparse_features = train_deepfm(db_path, cls_order_path, cls_food_path, cls_user_path, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, 
                                                                                                             dnn_activation, dnn_dropout, deepfm_learner, lr, patience, batch_size, num_epochs, validation_split)
    
    deepfm_user_food_df, deepfm_group_makers_df, deepfm_user_makers_df = predict_deepfm(feature_named_tuple, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, dnn_activation, dnn_dropout, 
                                                                                        model_out_file, deepfm_learner, lr, db_path, user_group_path, food_makers_path, user2id, item2id, users_df, food_df,
                                                                                        sparse_features, save_path='DeepFM/outputs', model_name='deepfm')

    ncf_user_food_df = pd.read_csv('NCF/outputs/ncf_user_food_score.csv')
    deepfm_user_food_df = pd.read_csv('DeepFM/outputs/deepfm_user_food_score.csv')
    ncf_group_makers_df = pd.read_csv('NCF/outputs/ncf_group_makers_score.csv')
    deepfm_group_makers_df = pd.read_csv('DeepFM/outputs/deepfm_group_makers_score.csv')
    ncf_user_makers_df = pd.read_csv('NCF/outputs/ncf_user_makers_score.csv')
    deepfm_user_makers_df = pd.read_csv('DeepFM/outputs/deepfm_user_makers_score.csv')

    def make_scores(ncf, deepfm):
        cols = list(ncf.columns)
        ncf.percentage = ncf.percentage*3
        ncf.rename(columns={'percentage':'percentage_n'}, inplace=True)
        deepfm.rename(columns={'percentage':'percentage_d'}, inplace=True)
        ncf.drop_duplicates(subset=cols[:2], inplace=True)
        deepfm.drop_duplicates(subset=cols[:2], inplace=True)
        sum_user_food = pd.concat([ncf, deepfm]).drop_duplicates().fillna(0).groupby(by=cols[:2]).sum().reset_index()
        sum_user_food['percentage'] = sum_user_food.percentage_d + sum_user_food.percentage_n
        sum_user_food.drop(columns=['percentage_n', 'percentage_d'], inplace=True)
        return sum_user_food

    user_foods_score = make_scores(ncf_user_food_df, deepfm_user_food_df)
    group_makers_score = make_scores(ncf_group_makers_df, deepfm_group_makers_df)
    user_makers_score = make_scores(ncf_user_makers_df, deepfm_user_makers_df)

    user_foods_score.to_csv('results/user_foods_score.csv', index=False)
    group_makers_score.to_csv('results/group_makers_score.csv', index=False)
    user_makers_score.to_csv('results/user_makers_score.csv', index=False)
    
    print('All three predictions are saved. Please check results file.')