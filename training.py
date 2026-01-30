import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from types import SimpleNamespace

def logistic_regression(train, target_train, val, target_val, test, target_test,random_state, kfold):
    model = LogisticRegression()
    stratified_kfold = StratifiedKFold(random_state=random_state, shuffle=True, n_splits=kfold)
    param = {
        'penalty': ['l2'],
        'C':[1.0, .1, .001],
        'class_weight': ['balanced', None],
        'solver' : ['lbfgs', 'liblinear'],
        'max_iter' :[1000]
    }
    grid_search = GridSearchCV(model, param_grid=param, scoring='f1', n_jobs=-1,cv=stratified_kfold,verbose=3)
    grid_search.fit(train, target_train)
    best_params = grid_search.best_params_
    print(f'best params for Logistic Regression are {best_params}')
    model = LogisticRegression(**best_params, random_state=random_state)
    model.fit(train,target_train)
    pred_train = model.predict(train)
    pred_val = model.predict(val)
    pred_test = model.predict(test)
    return SimpleNamespace(best_params = best_params,pred_train = pred_train, pred_val=pred_val, pred_test=pred_test,model_name =  'logistic regression', model = model)

def neural_network(train,target_train,val, target_val,test, target_test,random_state, kfold):
    model = MLPClassifier()
    params = {
        'hidden_layer_sizes':[
            # (12, 8, 2),
            # (16, 8,2),
            # (32, 16, 8),
            (64,32, 16),
            # (64, 32, 16, 8),
            # (128, 64)
            # (128, 64)
        ],
        'activation':['tanh'],
        'solver': ['sgd'],
        'alpha':[1],
        'max_iter':[1500],       
        'random_state':[random_state],
        'batch_size':[512],
        'learning_rate_init':[.001],
        'learning_rate':['adaptive'],
        'verbose':[2],
    }
 
    stratified_kfold = StratifiedKFold(random_state=random_state, shuffle=True, n_splits=kfold)
    gridsearch = GridSearchCV(param_grid=params, scoring='f1',estimator=model, n_jobs=-1, cv=stratified_kfold)
    gridsearch.fit(train,target_train)
    best_params = gridsearch.best_params_
    model = MLPClassifier(**best_params)
    model.fit(train,target_train)
    pred_train = model.predict(train)
    pred_val = model.predict(val)
    pred_test= model.predict(test)  
    return SimpleNamespace(best_params = best_params,pred_train = pred_train, pred_val=pred_val, pred_test=pred_test,model_name =  'NN', model = model)


def random_forests(train,target_train, val, target_val, test, target_test, random_state, kfold):
    model = RandomForestClassifier()
    params = {
        'n_estimators':[100],
        'max_depth':[15],
        'min_samples_split':[15],
        'min_samples_leaf':[7],
        'max_features':['log2'],
        'verbose':[2],
        'n_jobs':[-1],
        'random_state':[random_state]
    }
    stratified_kfold = StratifiedKFold(shuffle=True, n_splits=kfold, random_state=random_state)
    gridsearch = GridSearchCV(param_grid=params, estimator=model, n_jobs=-1, scoring='f1',cv=stratified_kfold)
    gridsearch.fit(train,target_train)
    best_params = gridsearch.best_params_
    model = RandomForestClassifier(**best_params)
    model.fit(train,target_train)
    pred_train = model.predict(train)
    pred_val = model.predict(val)
    pred_test = model.predict(test)
    return SimpleNamespace(best_params = best_params,pred_train = pred_train, pred_val=pred_val, pred_test=pred_test,model_name =  'Randomforests', model = model)

def voting_classifier(train,target_train, val, target_val, test, target_test, random_state, kfold):
    clf1 = LogisticRegression(
        penalty= 'l2',
        C=.1,
        max_iter=1000,
        solver='liblinear',
        random_state=random_state,
        n_jobs=-1
    )
    clf2 = MLPClassifier(
        solver='sgd',
        activation='tanh',
        hidden_layer_sizes=(64, 32, 16),
        alpha=1,
        learning_rate='adaptive',
        learning_rate_init=.001,
        batch_size=512,
        max_iter=1500,
        random_state=random_state
    )
    clf3 = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=15,
        min_samples_leaf=7,
        max_features='log2',
        random_state=random_state,
        n_jobs=-1
    )

    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('nn', clf2), ('rf', clf3)],
        voting='hard'
    )
    voting_clf.fit(train,target_train)
    pred_train = voting_clf.predict(train)
    pred_val = voting_clf.predict(val)
    pred_test = voting_clf.predict(test)
    return SimpleNamespace(pred_train = pred_train,pred_val=pred_val,pred_test=pred_test,model_name = "Voting Classifier",model = voting_clf)


def pred_with_threshold(model, train, target_train,val, target_val,test, target_test):
    pred_train_propa = model.predict_proba(train)[:,1]
    pred_val_propa = model.predict_proba(val)[:,1]
    pred_test_propa = model.predict_proba(test)[:,1]
    
    precision_val, recall_val, thresholds = precision_recall_curve(target_val,pred_val_propa)
    f1_scores = (2 * precision_val * recall_val) / (precision_val + recall_val)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    pred_train_opt = (pred_train_propa >= optimal_threshold).astype(int)
    pred_val_opt = (pred_val_propa >= optimal_threshold).astype(int)
    pred_test_opt = (pred_test_propa >= optimal_threshold).astype(int)
    return SimpleNamespace(pred_train_opt = pred_train_opt, pred_val_opt = pred_val_opt,pred_test_opt = pred_test_opt, optimal_threshold= optimal_threshold)