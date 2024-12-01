import numpy as np
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

"""
fixed: 2024_12_01

"""
class InfluenceBalancedXGB:
    def __init__(self, n_estimator=100, first_turn=2, metrics="f1", use_hessian=False):
        self.use_hessian = use_hessian
        self.n_estimator=n_estimator
        self.first_turn = first_turn
        if metrics in ["f1", "macro-precision", "g-mean"]:
            self.metrics = metrics
        else:
            raise ValueError("metrics is incorrect")
        
    def calc_weighted_accuracy(self, dtest, bst):
        y = dtest.get_label()
        y_pred = np.round(bst.predict(dtest))
        score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[1])
        return score

    def calc_recall(self, dtest, bst):
        y = dtest.get_label()
        y_pred = np.round(bst.predict(dtest))
        score = precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[1][1]
        return score

    def score(self, dtest):
        y = dtest.get_label()
        y_pred = np.round(self.bst.predict(dtest))
        if self.metrics == "precision":
            score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[0])
        elif self.metrics == "g-mean":
            t = precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[1]
            score = np.sqrt(t[0]*t[1])
        else:
            score = (precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[2][1])
        # print(score)
        return score
    def calc_weight(self, bst, dtrain):
        y = dtrain.get_label()
        lamda = self.params['reg_lambda']

        prob = bst.predict(dtrain)
        leaf_indices = bst.predict(dtrain, pred_leaf=True)
        if len(leaf_indices.shape) > 1:
            leaf_indices = leaf_indices[:, -1]
        results = {}
        for p, idx in zip(prob, leaf_indices):
            if idx in results:
                results[int(idx)] += p*(1-p)
            else:
                results[int(idx)] = p*(1-p)
        hessian = pd.Series(leaf_indices).map(results).values
        self.hessians.append(hessian)
        ib = abs(prob - y) / (hessian + lamda + 1e-5)

        if self.use_hessian:
            weight = (hessian + lamda) / (abs(prob - y)+1e-5)
        else:
            weight = 1 / (abs(prob - y)+1e-5)
        return weight, ib
         
    def fit(self, dtrain, dtest, params):
        self.models=[]
        self.hessians = []
        eta = params["eta"]
        params["eta"] = 1
        self.params = params
        X_train = dtrain.get_data().toarray()
        y_train = dtrain.get_label()

        p = y_train.mean()
        w = 0.5*(y_train/p + (1-y_train)/(1-p))
        dtrain_re = xgb.DMatrix(X_train, y_train, weights=w)
        # 初期モデルをトレーニング
        params['base_score'] = 0.8
        bst = xgb.train(params, dtrain_re, 
                        num_boost_round=self.first_turn)        
        self.bst = bst
        score = self.score(dtest=dtest)
        self.test_scores= [score for i in range(self.first_turn)]
        params["eta"] = eta

        bst2 = None
        params.pop('base_score')
        for i in range(self.n_estimator):
            # ここで新しい重みを計算 (例としてランダムに重みを変更)
            weights, ib = self.calc_weight(bst, dtrain)
            # 新しいDMatrixを作成
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
            # 以前のモデルから学習を再開
            # if i == 0:
            #     bst2 = xgb.train(params, dtrain, num_boost_round=1, 
            #                     xgb_model=bst,verbose_eval=False)
            # else:
            
            bst = xgb.train(params, dtrain, num_boost_round=1, 
                                xgb_model=bst,verbose_eval=False)
            self.bst=bst
            test_score = self.score(dtest=dtest)
            self.test_scores.append(test_score)

        ndarray = np.array(self.test_scores)
        max_idx = int(np.where(abs(max(ndarray)-ndarray)<1e-10)[0][0])
        self.stop_round=max_idx

        # print(max_idx)
        # for i in range(max_idx):
        #     # ここで新しい重みを計算 (例としてランダムに重みを変更)
        #     weights, ib = self.calc_weight(bst, dtrain)
        #     # 新しいDMatrixを作成
        #     dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        #     # 以前のモデルから学習を再開
            
        #     bst = xgb.train(params, dtrain, num_boost_round=1, 
        #                         xgb_model=bst,verbose_eval=False)
        # self.bst = bst



    def predict(self, dtest):
         idx = self.stop_round
         return self.bst.predict(dtest, iteration_range=(0, idx+1))