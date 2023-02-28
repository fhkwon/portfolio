import pandas as pd
import os

from .deepfm_model import DeepFM
from .preprocessing import fill_users
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from multi import compile_model, make_predict_original_unavailable, make_input_ids, make_leftovers, save_predictions


def predict_deepfm(feature_named_tuple, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, dnn_activation, dnn_dropout, 
                model_load_file, learner,lr, db_path, user_group_path, food_makers_path, user2id, item2id, users_df, food_df,
                sparse_features, save_path, model_name):

    new_model = DeepFM(feature_named_tuple, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, dnn_activation, dnn_dropout)
    new_model.load_weights(model_load_file)
    new_model = compile_model(new_model, learner, lr)
    
    available_group_user = pd.read_csv(os.path.join(db_path, user_group_path)).drop(columns=['Unnamed: 0'])
    available_users = list(available_group_user.UserId.values)
    available_food_makers = pd.read_csv(os.path.join(db_path, food_makers_path)).drop(columns=['Unnamed: 0'])
    available_foods = list(available_food_makers.FoodId.values)
    available_makers = list(available_food_makers.MakersId.values)
    food_makers_df = pd.DataFrame({'FoodId': available_foods, 'MakersId': available_makers})

    unavailable_users, predict_users, user_ids = make_predict_original_unavailable(available_users, user2id)
    unavailable_foods, predict_foods, food_ids = make_predict_original_unavailable(available_foods, item2id)

    pred_users,pred_foods = make_input_ids(predict_users, predict_foods)
    users,foods = make_input_ids(user_ids, food_ids)

    cur = pd.DataFrame([users, foods, pred_users, pred_foods]).T
    cur.columns = ['UserId', 'FoodId', 'pred_UserId', 'pred_FoodId']

    cur = cur.merge(users_df, how="left")
    cur = cur.merge(food_df, how="left")
    cur = cur.merge(available_group_user, how="left")

    null_columns = are_foodtags_notnull(food_df, cur)
    if len(null_columns)> 0:
        print(f'Check these columns : {null_columns}')
    cur, _ = fill_users(users_df, cur)
    cur.rename(columns = {'UserId':"original_UserId", 'FoodId':'original_FoodId', 'pred_UserId':'UserId', 'pred_FoodId':'FoodId'}, inplace=True)
    # cur.to_csv('cur_user_food.csv', index=False)
    pred_input = {name: cur[name] for name in sparse_features}

    newpred = new_model.predict(pred_input)
    newpreds = newpred.flatten()
    left_cur = make_leftovers(unavailable_users, unavailable_foods, user_ids, food_ids)
    user_food_df, group_makers_df, user_makers_df = save_predictions(users, foods, newpreds, available_group_user, food_makers_df, left_cur, save_path, model_name)

    return user_food_df, group_makers_df, user_makers_df

def are_foodtags_notnull(food_df, df):
    food_cols = list(food_df.columns)
    food_cols.remove('FoodId')
    null_dict = df[food_cols].isna().sum().to_dict()
    null_columns = [col for col, val in null_dict.items() if val >0]
    if df[food_cols].isna().sum().sum() > 0:
        raise ValueError("There are null values in food tags, therfore the model cannot predict")
    else:
        print("All food tags are notnull.")
    return null_columns


