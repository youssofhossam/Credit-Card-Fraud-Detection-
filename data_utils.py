import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier

random_state = 17
k_fold = 5
def load_data():
    train_df = pd.read_csv('Dataset/train.csv')
    test_df = pd.read_csv('Dataset/test.csv')
    val_df = pd.read_csv('Dataset/val.csv')
    train_val_df = pd.read_csv('Dataset/trainval.csv')
    train_df = train_df.drop_duplicates()

    target_train_df = train_df['Class']
    target_val_df = val_df['Class']
    target_test_df = test_df['Class']

    train_df = train_df.drop(columns=['Class'])
    val_df = val_df.drop(columns=['Class'])
    test_df = test_df.drop(columns=['Class'])
    return train_df,val_df,test_df,train_val_df,target_train_df,target_val_df,target_test_df

def handle_features(train_df, val_df, test_df):
    new_train_df = train_df
    new_val_df = val_df
    new_test_df = test_df
    new_train_df['log_amount'] = new_train_df['Amount']
    new_val_df['log_amount'] = new_val_df['Amount']
    new_test_df['log_amount'] = new_test_df['Amount']

    new_train_df = new_train_df.drop(columns=['Amount'])
    new_val_df = new_val_df.drop(columns=['Amount'])
    new_test_df = new_test_df.drop(columns=['Amount'])
    return new_train_df,new_val_df,new_test_df, 1

# scale
def scale_data(choice, train, val, test):
    if(choice == 'minmax'):
        scaler = MinMaxScaler()
    elif(choice == 'standard'):
        scaler = StandardScaler()
    elif(choice == 'robust'):
        scaler = RobustScaler()
    elif(choice == None):
        choice = 'None'
        return train, val, test, choice

    new_train = scaler.fit_transform(train)
    new_val = scaler.transform(val)
    new_test = scaler.transform(test)
    return new_train, new_val,new_test, choice

### Handle imbalance
def standard_smote(sampling_strategy, random_state, train, target_train):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    new_train, new_target_train = smote.fit_resample(train, target_train)
    return new_train, new_target_train

def borderline_smote(sampling_strategy, random_state, train, target_train):
    bordersmote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    new_train, new_target_train = bordersmote.fit_resample(train, target_train)
    return new_train, new_target_train


def smote_tomek(sampling_strategy, random_state, train, target_train):
    smotomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    new_train, new_target_train = smotomek.fit_resample(train, target_train)
    return new_train, new_target_train

def under_sampling(sampling_strategy,version,train,target_train):
    undersampling = NearMiss(sampling_strategy=sampling_strategy, version=version)
    new_train,new_target_train = undersampling.fit_resample(train,target_train)
    # version 2 is the best
    return new_train,new_target_train

def balance_data(choice_smote, sampling_strategy, train, target_train,version):
    if(choice_smote == 'smote'):
        new_train , new_target_train = standard_smote(random_state=random_state, sampling_strategy=sampling_strategy,train= train,target_train= target_train)
    elif(choice_smote == 'bsmote'):
        new_train , new_target_train = borderline_smote(random_state=random_state, sampling_strategy=sampling_strategy,train= train,target_train= target_train)
    elif(choice_smote == 'tsmote'):
        new_train , new_target_train = smote_tomek(random_state=random_state, sampling_strategy=sampling_strategy,train= train,target_train= target_train)
    elif(choice_smote == 'under'):
        new_train , new_target_train = under_sampling(sampling_strategy=sampling_strategy,train= train,target_train= target_train,version=version)
    else:
        new_train, new_target_train = train, target_train
    # if(choice_focal == 'focal'):
    #     return     
    return new_train, new_target_train, choice_smote

# convert
def convert_numpy(train_df, val_df,test_df, target_train_df, target_val_df, target_test_df):
    train = train_df.to_numpy()
    val = val_df.to_numpy()
    test = test_df.to_numpy()
    target_train = target_train_df.to_numpy()
    target_val = target_val_df.to_numpy()
    target_test = target_test_df.to_numpy()
    return train,val,test,target_train,target_val,target_test

# Modeling
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
    return best_params,pred_train, pred_val, pred_test, 'logistic regression', model

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
    return best_params, pred_train, pred_val,pred_test, "NN",model


def random_forests(train,target_train, val, target_val, test, target_test, random_state, kfold):
    model = RandomForestClassifier()
    params = {
        'n_estimators':[100],
        'max_depth':[15],
        'min_samples_split':[12],
        'min_samples_leaf':[3],
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
    return best_params, pred_train,pred_val,pred_test,"Randomforests", model

# use optimal threshold

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
    return pred_train_opt, pred_val_opt,pred_test_opt, optimal_threshold

# Evaluation
def evaluate(pred_train, pred_val,pred_test, target_train,target_val,target_test):
    train_rep = classification_report(target_train, pred_train)    
    val_rep = classification_report(target_val, pred_val)    
    test_rep = classification_report(target_test, pred_test)    
    return train_rep, val_rep,test_rep

def eveluate_threshold(pred_train_opt, pred_val_opt,pred_test_opt, target_train,target_val,target_test):
    train_rep = classification_report(target_train, pred_train_opt)    
    val_rep = classification_report(target_val, pred_val_opt)    
    test_rep = classification_report(target_test, pred_test_opt)    
    return train_rep, val_rep,test_rep

# save answer
def save_rep(model_name, processor_name, handeld_features, params):
    with open('classification_reports.txt', 'a') as f:
        f.write(f'model: {model_name}, scale: {processor_name}, feature engineering: {handeld_features} ,model params: {params} \n')
        f.write(f'oversampling type : {smote_type} \n')
        f.write(train_rep)
        f.write(val_rep)
        f.write(test_rep)
        f.write('################################################################\n')

if(__name__ == '__main__'):
    train_df,val_df,test_df,train_val_df,target_train_df,target_val_df,target_test_df = load_data()
    handeld_features = 0
    train_df,val_df,test_df, handeld_features = handle_features(train_df,val_df,test_df)
    smote_type = None
    train,val,test,target_train,target_val,target_test = convert_numpy(train_df,val_df,test_df,target_train_df,target_val_df,target_test_df)
    train,val,test, processor_name = scale_data('standard', train,val,test)
    train,target_train, smote_type = balance_data('tsmote', .01, train, target_train,version=2) 
    print(train.shape)
    # Train with Logistic Regression
    # best_params_logistic, pred_train, pred_val, pred_test, model_name, model = logistic_regression(train, target_train,val, target_val,test,target_test,random_state=random_state,kfold=k_fold)
    # pred_train_opt, pred_val_opt,pred_test_opt, optimal_threshold = pred_with_threshold(model = model,train=train,val=val,test=test,target_train=target_train, target_val=target_val,target_test=target_test ) 
    # print(best_params_logistic)
    # train with NN
    # best_params_nn, pred_train, pred_val, pred_test, model_name, model = neural_network(train, target_train,val, target_val,test,target_test,random_state=random_state,kfold=k_fold)
    # pred_train_opt, pred_val_opt,pred_test_opt, optimal_threshold = pred_with_threshold(model = model,train=train,val=val,test=test,target_train=target_train, target_val=target_val,target_test=target_test ) 
    # print(best_params_nn)

    # train with Randomforests
    best_params_forests, pred_train, pred_val, pred_test, model_name, model = random_forests(train, target_train,val, target_val,test,target_test,random_state=random_state,kfold=k_fold)
    pred_train_opt, pred_val_opt,pred_test_opt, optimal_threshold = pred_with_threshold(model = model,train=train,val=val,test=test,target_train=target_train, target_val=target_val,target_test=target_test ) 
    print(best_params_forests)

    # Evaluate
    # train_rep, val_rep,test_rep = evaluate(pred_train, pred_val,pred_test, target_train,target_val,target_test)
    # evaluate for optimal threhold
    train_rep, val_rep,test_rep = eveluate_threshold(pred_train_opt, pred_val_opt,pred_test_opt, target_train,target_val,target_test)
    save_rep(model_name=model_name, handeld_features=handeld_features, processor_name= processor_name, params=best_params_forests) 
    print(train_rep, val_rep, test_rep)
    # x = 5, y = 0
    # if(x > 0 or x < 0 or y == 0):
        