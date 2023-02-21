import os
import pandas as pd
import numpy as np

from datetime import datetime
from time import time

from .preprocessing import preprocess_data, train_test_split, make_train_lists, make_test_lists
from .ncf_model import NCF, EarlyStopping
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from multi import compile_model
from .eval import evaluate_model

def train_ncf(db_path, order_path, num_ng_test, num_neg, embedding_size, 
              layers, reg_layers, reg_mf, learner, lr, num_epochs, batch_size, verbose, out, topK):
    data = pd.read_csv(os.path.join(db_path, order_path), header=0, names=['GroupId','UserId','MakersId','FoodId','orderdate'])
    data.drop(columns=['GroupId','MakersId'])
    data['label'] = 1
    data, user2id, item2id, item_pool, num_users, num_items = preprocess_data(data)
    train_ratings, test_ratings = train_test_split(data, item_pool, num_ng_test, num_neg)
    user_input, item_input, labels = make_train_lists(train_ratings, num_neg)
    ts_ui, neg_items = make_test_lists(test_ratings)

    model = NCF(num_users, num_items, embedding_size, layers, reg_layers, reg_mf)

    model = compile_model(model, learner, lr)

    early_stopping = EarlyStopping(patience = 5, verbose = True)

    # Init performance
    (hits, ndcgs) = evaluate_model(model, ts_ui, neg_items, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    
    for epoch in range(num_epochs):
        t1 = time()
        hist = model.fit([np.array(user_input), np.array(item_input)], 
                            np.array(labels), 
                            batch_size=batch_size, epochs=1, verbose=2, shuffle=True)
        t2 = time()
        
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, ts_ui, neg_items, topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                    % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch           

        early_stopping(hr, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if out > 0:
        datentime = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        model_out_file = f'Pretrain/NeuMF_{datentime}.h5'
        model.save_weights(model_out_file, overwrite=True)
        print("The best NeuMF model is saved to %s" %(model_out_file))
    
    return num_users, num_items, model_out_file, user2id, item2id