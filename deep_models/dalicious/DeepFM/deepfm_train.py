import os
import pandas as pd

from datetime import datetime
from tensorflow import keras

from .preprocessing import drop_foodtag,fill_users, split_xy, SparseFeat
from .deepfm_model import DeepFM
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from multi import compile_model


def train_deepfm(db_path, cls_order_path, cls_food_path, cls_user_path, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, 
                 dnn_activation, dnn_dropout, learner, lr, patience, batch_size, num_epochs, validation_split):
    data = pd.read_csv(os.path.join(db_path, cls_order_path),  encoding='cp949', engine='python', index_col='Unnamed: 0' )
    COL_NAME = ['OrderId', 'UserId', 'Created', 'selected_FoodId', 'GroupId', 'FoodId', 'selected']
    data.columns = COL_NAME
    unneccessary_cols = ['selected_FoodId', 'Created']
    data.drop(columns=unneccessary_cols, inplace=True)
    food_df = pd.read_csv(os.path.join(db_path, cls_food_path), engine='python', index_col='Unnamed: 0' )
    food_df.rename(columns={'요리스타일_한식': 'cook_korean', '요리스타일_분식': 'cook_bunsik', '요리스타일_중식': 'cook_chinese', '요리스타일_일식': 'cook_japanese', 
                            '요리스타일_양식': 'cook_western', '요리스타일_동남아': 'cook_eastasia', '요리스타일_인도': 'cook_india', '국가_한국':'nation_korea', '국가_중국': 'nation_china', 
                            '국가_홍콩': 'nation_honkong', '국가_대만': 'nation_taiwan', '국가_일본': 'nation_japan', '국가_미국': 'nation_america', '국가_멕시코': 'nation_mexico', 
                            '국가_스페인': 'nation_spain', '국가_지중해': 'nation_mediterranean', '국가_이탈리아': 'nation_italy', '국가_프랑스': 'nation_france', '국가_베트남': 'nation_vietnam', 
                            '국가_태국':'nation_thailand', '식품유형_떡류': 'ftype_ricecake', '식품유형_밥류':'ftype_rice', '식품유형_빵류':'ftype_bread', '식품유형_면류':'fype_noodle', 
                            '식품유형_김치류': 'ftype_kimchi', '식품유형_회류': 'ftype_rawfish', '식품유형_젓갈류': 'ftype_jeotgal', '식품유형_장아찌류': 'ftype_jangajji', '식품유형_양념류':'ftype_seasoning', 
                            '식품유형_유제품류': 'ftype_diary', '식품유형_음료': 'ftype_drinks', '식품유형_차류':'ftype_teas', '식품유형_과일류':  'fype_fruits', '식품유형_죽류':'ftype_porridge', 
                            '식품유형_국/탕류':'ftype_soup', '식품유형_찌개류':'ftype_stew', '식품유형_찜류':'ftype_jjim', '식품유형_구이류': 'ftype_roasted', '식품유형_전/부침류':'ftype_buchim', 
                            '식품유형_샐러드': 'ftype_salad', '주재료_소고기': 'ing_beef', '주재료_돼지고기': 'ing_pork', '주재료_닭고기':'ing_chicken', '주재료_오리고기':'ing_duck', '주재료_양고기': 'ing_lamb', 
                            '주재료_생선류': 'ing_fish', '주재료_조개류':'ing_clams', '주재료_갑각류': 'ing_crustacean', '주재료_연체류': 'ing_mollusks', '주재료_채소류':'ing_vegi', 
                            '주재료_곡류': 'ing_grain', '주재료_난류':'ing_eggs', '주재료_버섯류':'ing_mushrooms', '조리법_볶다':'method_panfry', '조리법_찌다':'method_steam', 
                            '조리법_튀기다':'method_fry', '조리법_굽다':'method_roast', '조리법_삶다':'method_simmer', '조리법_끓이다':'method_boil', '조리법_베이킹':'method_baking', 
                            '조리법_졸이다':'method_boildown', '조리법_절이다':'method_pickle', '온도_상온':'temp_room', '온도_시원함':'temp_cool', '온도_차가움':'temp_cold', '온도_따뜻함':'temp_warm', 
                            '온도_뜨거움':'temp_hot', '맛_단맛':'flav_sweet', '맛_매운맛':'flav_spicy', '맛_짠맛':'flav_salty', '맛_쓴맛':'flav_bitter', '맛_신맛':'flav_sour', '맛_감칠맛':'flav_gmachil', 
                            '메뉴성격_든든한':'char_full', '메뉴성격_가벼운':'char_light', '메뉴성격_헤비한':'char_heavy', '메뉴성격_간편한':'char_conv', '메뉴성격_건강한':'char_healthy', '메뉴성격_기름진':'char_greasy', 
                            '메뉴성격_얼얼한':'char_numbing', '메뉴성격_단짠단짠':'char_sweetsalty', '메뉴성격_새콤달콤':'char_sweetsour', '메뉴성격_싱거운':'char_bland', '메뉴성격_불맛':'char_grill', 
                            '메뉴성격_느끼한':'char_oily', '메뉴성격_쫄깃한':'char_chewy', '메뉴성격_바삭한':'char_crispy', '메뉴성격_아삭한':'char_crunch', '메뉴성격_신선한':'char_fresh', '메뉴성격_순한':'char_mild', 
                            '메뉴성격_촉촉한':'char_moist', '메뉴성격_톡쏘는':'char_tangy', '메뉴성격_구수한':'char_gusu', '메뉴성격_버터향이 나는':'char_buttery', '알레르기체크_우유':'allergy_milk', 
                            '알레르기체크_메밀':'allergy_buckwheat', '알레르기체크_땅콩':'allergy_peanut', '알레르기체크_대두':'allergy_soybean', '알레르기체크_밀':'allergy_wheat', '알레르기체크_고등어':'allergy_mackeral', 
                            '알레르기체크_게':'allergy_crab', '알레르기체크_새우':'allergy_shrimp', '알레르기체크_돼지고기':'allergy_pork', '알레르기체크_아황산류':'allergy_sulfurousacid', '알레르기체크_토마토':'allergy_tomato', 
                            '알레르기체크_호두':'allergy_walnut', '알레르기체크_닭고기':'allergy_chicken', '알레르기체크_알류':'allergy_eggs', '알레르기체크_쇠고기':'allergy_beef', '알레르기체크_오징어':'allergy_squid', 
                            '알레르기체크_조개류':'allergy_clams', '알레르기체크_잣':'allergy_pinenut', '특이식성_채식':'pref_vegi', '특이식성_글루텐프리':'pref_glutenfree', '특이식성_키토제닉':'pref_keto', 
                            '특이식성_다이어트':'pref_diet', '메뉴 제공방식_도시락':'served_dosirak', '메뉴 제공방식_코스':'served_course', '메뉴 제공방식_뷔페':'served_buffet', 
                            '메뉴 제공방식_메인메뉴':'served_main', '메뉴 제공방식_디저트':'served_dessert', '메뉴 제공방식_박스 케이터링':'served_box', '매움정도_1':'spicy_1', 
                            '매움정도_2':'spicy_2', '매움정도_3':'spicy_3'}, inplace=True)
    users_df = pd.read_csv(os.path.join(db_path, cls_user_path),  encoding='cp949', engine='python', index_col='Unnamed: 0')
    data = data.merge(users_df, how="left")
    data = data.merge(food_df, how="left")
    data.set_index('OrderId', inplace=True)
    
    data = drop_foodtag(food_df, data)
    data, max_values = fill_users(users_df, data)
    user2id = {w: i for i, w in enumerate(list(set(data['UserId'])))}
    item2id = {w: i for i, w in enumerate(list(set(data['FoodId'])))}
    sparse_features, target = split_xy(data)
    feature_named_tuple = [SparseFeat(feat, int(data[feat].values.max()) + 1, embedding_dim=8) for feat in sparse_features]
    model_input = {name: data[name] for name in sparse_features}

    model = DeepFM(feature_named_tuple, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, dnn_activation, dnn_dropout)
    model = compile_model(model, learner, lr)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit(model_input, data[target].values,
                        batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=validation_split, callbacks=[early_stopping_callback])
    datentime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_out_file = f'DeepFM/Pretrain/DeepFM_{datentime}.h5'
    model.save_weights(model_out_file, overwrite=True)

    return model_out_file, user2id, item2id, users_df, food_df, feature_named_tuple, sparse_features