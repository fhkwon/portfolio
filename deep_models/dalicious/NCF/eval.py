import math
import heapq
import numpy as np

def evaluate_model(model, ts_ui, neg_items, topK):
    hits, ndcgs = [],[]
    for idx in range(len(ts_ui)):
        (hr,ndcg) = eval_one_rating(idx, ts_ui, neg_items, model, topK)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx, ts_ui, neg_items, model, topK):
    rating = ts_ui[idx]
    u = rating[0]
    gtItem = rating[1]
    items = neg_items[idx]
    items.append(gtItem)
    users = np.full(len(items), u, dtype = 'int32')
    predictions = model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    map_item_score = {}
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get) 
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

