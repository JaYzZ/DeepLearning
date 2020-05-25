# -*- coding: utf-8 -*-

import os
import sys
import traceback
import datetime
import argparse
import json
import numpy as np
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def data_config():
    parser = argparse.ArgumentParser()
    # Module Args
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--ngram_file', type=str, required=True)
    parser.add_argument('--ngram_name', type=str, default='ngram.txt')
    parser.add_argument('--output_model', type=str, required=True)
    parser.add_argument('--data_view', type=str, default='content')
    parser.add_argument('--label_view', type=str, default='label')
    parser.add_argument('--positive_class_name', type=str, default='oof')
    parser.add_argument('--test_ratio', type=float, default=0.3)

    # XGB Args
    parser.add_argument('--num_round', type=int, default=None)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_child_weight', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--subsample', type=float, default=None)
    parser.add_argument('--colsample_bytree', type=float, default=None)
    parser.add_argument('--scale_pos_weight', type=float, default=None)
    parser.add_argument('--reg_alpha', type=float, default=None)
    parser.add_argument('--reg_lambda', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    return vars(parser.parse_args())


def log(info):
    """Log the not compliant info"""
    print("SystemLog: [UTC: %s] %s" % (datetime.datetime.utcnow(), info), flush=True)


def folderexpand(folder, filename=None):
    if os.path.isfile(folder):
        return [folder]

    if filename is not None:
        return [os.path.join(folder, filename)] 

    files_list = []
    # only one level search
    for root, _, files in os.walk(folder):
        for f in files:
            file_path = os.path.join(root, f)
            files_list.append(file_path)
    return files_list


def model_cv(model, X, y, cv_folds=5, early_stopping_rounds=50, seed=0):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                    metrics='auc', seed=seed, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(early_stopping_rounds)
       ])
    num_round_best = cvresult.shape[0] - 1
    log(f'Best round num: {num_round_best}')
    return num_round_best


def gridsearch_cv(model, test_param, X, y, cv=5):
    gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='roc_auc', n_jobs=-1, iid=False, cv=cv)
    gsearch.fit(X, y)
    log(f'CV Results: {gsearch.cv_results_}')
    log(f'Best Params: {gsearch.best_params_}')
    log(f'Best Score: {gsearch.best_score_}')
    return gsearch.best_params_


def latent_visualization(x, y):
    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=100).fit_transform(x.A)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
    big_groups = list(sorted(set(y)))
    for big_group in big_groups:
        for group in [big_group, big_group + 100]:
            a, b = [], []
            for j, label in enumerate(y):
                if label == group:
                    a.append(tsne[j][0])
                    b.append(tsne[j][1])
            color = colors[int(group % 100)]
            marker = 'x' if group < 100 else 'o'
            size = 1 if group < 100 else 27
            ax.scatter(a, b, color=color, marker=marker, s=size)
            plt.axis('off')
    plt.show()


def xgb_tuning(x, x_test, y, y_test, opt):
    # latent_visualization(x, y)    

    # default hyper params
    num_round = 5000
    seed = 0
    max_depth = 3
    min_child_weight = 7
    gamma = 0
    subsample = 0.8
    colsample_bytree = 0.8
    scale_pos_weight = 1
    reg_alpha = 1
    reg_lambda = 1e-5
    learning_rate = 0.1
    
    def init_model():
        return XGBClassifier(learning_rate=learning_rate, n_estimators=num_round, max_depth=max_depth,
                    min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, objective='binary:logistic',
                    nthread=4, scale_pos_weight=scale_pos_weight, seed=seed)

    model = init_model()

    # tune num_round
    if 'num_round' in opt.keys() and opt['num_round'] is not None:
        num_round = opt['num_round']
        log('Preset num_round')
    else:
        num_round = model_cv(model, x, y)
        log('Finish Tuning num_round')
    model = init_model()

    # tune max_depth & min_child_weight
    if ('max_depth' in opt.keys() and opt['max_depth'] is not None) or ('min_child_weight' in opt.keys() and opt['min_child_weight'] is not None):
        max_depth = opt['max_depth']
        min_child_weight = opt['min_child_weight']
        log('Preset max_depth & min_child_weight')
    else:
        param_test1 = {
            'max_depth': range(3, 11, 1),
            'min_child_weight': range(1, 11, 1)
        }
        best_param1 = gridsearch_cv(model, param_test1, x, y)
        max_depth = best_param1['max_depth']
        min_child_weight = best_param1['min_child_weight']
        log('Finish Tuning max_depth & min_child_weight')
    model = init_model()

    # tune gamma
    if 'gamma' in opt.keys() and opt['gamma'] is not None:
        gamma = opt['gamma']
        log('Preset gamma')
    else:
        param_test2 = {
            'gamma': [i / 100.0 for i in range(0, 50)]
        }
        best_param2 = gridsearch_cv(model, param_test2, x, y)
        gamma = best_param2['gamma']
        log('Finish Tuning gamma')
    model = init_model()

    # tune subsample & colsample_bytree
    if ('subsample' in opt.keys() and opt['subsample'] is not None) or ('colsample_bytree' in opt.keys() and opt['colsample_bytree'] is not None):
        subsample = opt['subsample']
        colsample_bytree = opt['colsample_bytree']
        log('Preset subsample & colsample_bytree')
    else:
        # Round 1
        param_test3 = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }
        best_param3 = gridsearch_cv(model, param_test3, x, y)
        subsample = best_param3['subsample']
        colsample_bytree = best_param3['colsample_bytree']
        model = init_model()
        # Round 2
        param_test3 = {
            'subsample': [i / 10.0 for i in range(int(subsample * 10 - 1), int(subsample * 10 + 1))],
            'colsample_bytree': [i / 10.0 for i in range(int(colsample_bytree * 10 - 1), int(colsample_bytree * 10 + 1))]
        }
        best_param3 = gridsearch_cv(model, param_test3, x, y)
        subsample = best_param3['subsample']
        colsample_bytree = best_param3['colsample_bytree']
        log('Finish Tuning subsample & colsample_bytree')
    model = init_model()
    

    # tune scale_pos_weight
    if 'scale_pos_weight' in opt.keys() and opt['scale_pos_weight'] is not None:
        scale_pos_weight = opt['scale_pos_weight']
        log('Preset scale_pos_weight')
    else:
        param_test4 = {
            'scale_pos_weight': [i for i in range(1, 10, 2)],
        }
        best_param4 = gridsearch_cv(model, param_test4, x, y)
        scale_pos_weight = best_param4['scale_pos_weight']
        log('Finish Tuning scale_pos_weight')
    model = init_model()

    # tune reg_alpha & reg_lambda
    if ('reg_alpha' in opt.keys() and opt['reg_alpha'] is not None) or ('reg_lambda' in opt.keys() and opt['reg_lambda'] is not None):
        subsample = opt['reg_lambda']
        colsample_bytree = opt['reg_lambda']
        log('Preset reg_alpha & reg_lambda')
    else:
        param_test5 = {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100, 1000],
            'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100, 1000]
        }
        best_param5 = gridsearch_cv(model, param_test5, x, y)
        reg_alpha = best_param5['reg_alpha']
        reg_lambda = best_param5['reg_lambda']
        log('Finish Tuning reg_alpha & reg_lambda')
    model = init_model()
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, stratify=y, test_size=opt['test_ratio'])
    model.fit(x_train, y_train)

    log('Validation:')
    y_pred_proba = model.predict_proba(x_val)
    for i in np.linspace(0, 1, 21):
        y_pred = [1 if p >= i else 0 for p in y_pred_proba[:, 1]]
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        log('Test Performance %s: threshold: %.2f, precision: %.4f, recall: %.4f, f1: %.4f' % ('XGBoost', i, precision, recall, f1))

    log('Testing:')
    y_pred_proba = model.predict_proba(x_test)
    for i in np.linspace(0, 1, 21):
        y_pred = [1 if p >= i else 0 for p in y_pred_proba[:, 1]]
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        log('Test Performance %s: threshold: %.2f, precision: %.4f, recall: %.4f, f1: %.4f' % ('XGBoost', i, precision, recall, f1))

    return model


def main():
    log("Executing Start...")
    opt = data_config()

    vocab = []
    with open(folderexpand(opt['ngram_file'], filename=opt['ngram_name'])[0], 'r', encoding='utf-8', errors='ignore') as reader:
        ngram_list = reader.readlines()
        for gram in ngram_list:
            gram = gram.split('\t')[0].strip('\n').replace('_', ' ')
            vocab.append(gram)

    log('Successfully Load NGram File.')

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for file_name in folderexpand(opt['input_data']):
        with open(file_name, 'r', encoding='utf-8', errors='ignore') as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                train_data.append(data[opt['data_view']])
                train_label.append(1 if (data[opt['label_view']] == opt['positive_class_name'] or data[opt['label_view']] == 1) else 0)

    log('Successfully Load Train Data.')

    for file_name in folderexpand(opt['test_data']):
        with open(file_name, 'r', encoding='utf-8', errors='ignore') as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                test_data.append(data[opt['data_view']])
                test_label.append(1 if (data[opt['label_view']] == opt['positive_class_name'] or data[opt['label_view']] == 1) else 0)
    
    log('Successfully Load Test Data.')

    # Vectorize Train-Validation Set
    vectorizer = CountVectorizer(vocabulary=vocab, stop_words=None, token_pattern=r'\b[^\d\W]+\b', ngram_range=(1, 3))
    X = vectorizer.fit_transform(train_data)
    X_1 = np.zeros(X.A.shape)
    X_1[np.nonzero(X.A)] = 1
    X = X_1
    y = train_label
    # X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=y, test_size=opt['test_ratio'])
    log('Successfully Vectorize Train Set.')

    # Vectorize Test Set
    X_test = vectorizer.transform(test_data)
    X_1 = np.zeros(X_test.A.shape)
    X_1[np.nonzero(X_test.A)] = 1
    X_test = X_1
    y_test = test_label
    log('Successfully Vectorize Test Set.')

    xgb_model = xgb_tuning(X, X_test, y, y_test, opt)
    joblib.dump(xgb_model, os.path.join(opt['output_model'], "XGB_model.joblib"))
    log("Successfully Output XGB Model.")


if __name__ == '__main__':
    try:
        main()
    except BaseException as exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        # log('ERROR: %s' % str(exception.__class__.__name__))
        # log('ERROR: %s' % str(exception))
        raise exception
