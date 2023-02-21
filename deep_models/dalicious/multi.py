import os
import pandas as pd
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

def compile_model(model, learner, lr):
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=lr), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=lr), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=lr), loss='binary_crossentropy')
    return model

def make_predict_original_unavailable(user_or_item, list2id):
    unavialables = []
    predict = []
    ids = []
    for num in user_or_item: 
        try:
            predict.append(list2id[num])
            ids.append(num)
        except KeyError:
            unavialables.append(num)
    return unavialables, predict, ids

def make_input_ids(user_ids, food_ids):
    users = []
    for i in user_ids:
        users.extend([i]*len(food_ids))
    foods = food_ids*len(user_ids)
    return users, foods

def make_leftovers(unavailable_users, unavailable_foods, user_ids, food_ids):
    left_users = []
    for i in unavailable_users:
        left_users.extend([i]*len(unavailable_foods))
        left_users.extend([i]*len(food_ids))
    for i in user_ids:
        left_users.extend([i]*len(unavailable_foods))
    left_foods = unavailable_foods*(len(unavailable_users)+len(user_ids))
    left_foods.extend(food_ids*(len(unavailable_users)))
    left_cur = pd.DataFrame([left_users, left_foods]).T
    left_cur.columns = ['UserId', 'FoodId']
    left_cur['percentage'] = 0
    return left_cur

def save_predictions(users, foods, newpreds, available_group_user, food_makers_df, left_cur, save_path, model_name):
    user_food_df = pd.DataFrame([users, foods, newpreds]).T
    user_food_df.columns = ['UserId', 'FoodId', 'percentage']
    user_food_df = pd.concat([user_food_df, left_cur])
    user_food_df = user_food_df.sort_values(by=['UserId','FoodId'])
    user_food_df.to_csv(os.path.join(save_path, f"{model_name}_user_food_score.csv"), index=False)

    user_group_food_df = user_food_df.merge(available_group_user, how='left')
    user_food_makers_group_df = user_group_food_df.merge(food_makers_df, how='left')

    group_makers_df = user_food_makers_group_df.groupby(by=['GroupId', 'MakersId']).mean().reset_index()
    group_makers_df.drop(columns=['UserId', 'FoodId'], inplace=True)
    group_makers_df = group_makers_df.sort_values(by=['GroupId','MakersId'])
    group_makers_df.to_csv(os.path.join(save_path, f"{model_name}_group_makers_score.csv"), index=False)

    user_makers_df = user_food_makers_group_df.groupby(by=['UserId', 'MakersId']).mean().reset_index()
    user_makers_df.drop(columns=['GroupId', 'FoodId'], inplace=True)
    user_makers_df = user_makers_df.sort_values(by=['UserId','MakersId'])
    user_makers_df.to_csv(os.path.join(save_path, f"{model_name}_user_makers_score.csv"), index=False)
    print(f'All three predictions are saved. Please check {save_path}')
    return user_food_df, group_makers_df, user_makers_df

def make_scores(ncf, deepfm):
    cols = list(ncf.columns)
    if len(ncf) != len(deepfm):
        print(f'The length of the elements of ncf and deepfm does not mathch:,     ncf: {len(ncf)}, deepfm:{len(deepfm)}')
    score_df = pd.DataFrame([ncf[cols[0]], ncf[cols[1]], ncf.percentage*3 + deepfm.percentage]).T
    score_df.columns = ['UserId', 'FoodId', 'score']
    score_df.fillna(0)
    return score_df