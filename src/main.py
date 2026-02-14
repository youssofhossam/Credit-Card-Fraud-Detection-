import seaborn as sns
import matplotlib.pyplot as plt
import evaluation
from sklearn.metrics import recall_score, precision_score, f1_score
from types import SimpleNamespace
from training import logistic_regression , neural_network, random_forests, voting_classifier, pred_with_threshold
from data_utils import convert_numpy, load_data, handle_features, scale_data, balance_data
import pandas as pd
from config_utils import load_params
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate_test
import joblib
import os
import numpy
random_state = 17
k_fold = 5
models = []
threshold_models = []

def run_models(train, val, test, target_train,target_val,target_test):
    lr = logistic_regression(train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test,random_state=random_state,kfold=k_fold)
    nn = neural_network(train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test,random_state=random_state,kfold=k_fold)
    rf = random_forests(train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test,random_state=random_state,kfold=k_fold)
    vc= voting_classifier(train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test,random_state=random_state,kfold=k_fold)
    models.append(lr)
    models.append(nn)
    models.append(rf)
    models.append(vc)
    lr_opt = pred_with_threshold(model=lr.model, train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test)
    nn_opt = pred_with_threshold(model=nn.model, train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test)
    rf_opt = pred_with_threshold(model=rf.model, train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test)
    threshold_models.append(lr_opt)
    threshold_models.append(nn_opt)
    threshold_models.append(rf_opt)
    return rf.model
def compare_models(models,train, val, test, target_train,target_val,target_test):
    precision = []
    recall = []
    f1 = []
    for model in models:
        precision.append(precision_score(model.pred_val, target_val))
        recall.append(recall_score(model.pred_val, target_val))
        f1.append(f1_score(model.pred_val, target_val))
    
    for model_opt in threshold_models:
        precision.append(precision_score(model_opt.pred_val_opt, target_val))
        recall.append(recall_score(model_opt.pred_val_opt, target_val))
        f1.append(f1_score(model_opt.pred_val_opt, target_val))
    data = {
        'Model': ['Logistic Regression', 
                  'Neural Network',
                  'Random Forest',
                  'Voting Classifier',
                  'Logistic Regression (opt. threshold)',
                  'Neural Network (opt. threshold)',
                  'Random Forest (opt. threshold)'],
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
    compare_df = pd.DataFrame(data)
    compare_df.set_index('Model' ,inplace=True)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(data=compare_df,annot=True, 
    cmap='viridis', 
    fmt='.2f')
    plt.title('Model Performance Comparison')
    plt.ylabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def run(model_name, test,target_test,train,target_train):
    saved_params = load_params(model_name)
    if(model_name == 'Logistic Regression'):
        model = LogisticRegression(**saved_params)
    elif(model_name == 'Neural Network'):
        model = MLPClassifier(**saved_params)
    elif(model_name == 'Random Forest'):
        model = RandomForestClassifier(**saved_params)
    elif(model_name == 'Voting classifier'):
        vc= voting_classifier(train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test,random_state=random_state,kfold=k_fold)
        print(evaluate_test(pred_test=vc.pred_test,target_test=target_test))
    else:
        print('This Model is not present')
    model.fit(train,target_train)
    pred_test = model.predict(test)
    print(evaluate_test(pred_test=pred_test,target_test=target_test))

def deploy_model(train,val,test,target_train,target_val,target_test, scaler):
    test_sample =test.sample(50, random_state)
    test_sample['Class']= target_test.loc[test_sample.index]
    test_sample.to_csv('test_sample.csv', index=False)
    print('test_sample saved')
    final_model =run_models(train,val,test,target_train,target_val,target_test)
    path = 'final_fraud_model.pkl'
    joblib.dump({
        'model': final_model,
        'scaler': scaler
    }, path)

   

def load_model(test_case, path):
    data = joblib.load(path)
    model = data['model']
    scaler = data['scaler']
    scaled_test_case = scaler.transform(test_case)
    print(evaluate_test(model=model, pred_test=scaled_test_case, train=train,val=val, target_train=target_train,target_val=target_val,target_test=target_test))

if (__name__ == '__main__'):
    train_df,val_df,test_df,train_val_df,target_train_df,target_val_df,target_test_df = load_data()
    handeld_features = 0
    train_df,val_df,test_df, handeld_features = handle_features(train_df,val_df,test_df)
    smote_type = None
    train,val,test,target_train,target_val,target_test = convert_numpy(train_df,val_df,test_df,target_train_df,target_val_df,target_test_df)
    train,val,test, processor_name,scaler = scale_data('standard', train,val,test)
    train,target_train, smote_type = balance_data('tsmote', .01, train, target_train,version=2) 

    path = 'final_fraud_model.pkl'
    
    # deploy_model(train,val,test,target_train,target_val,target_test, scaler = scaler)
    # test_case =test[2]
    # load_model(test_case=[test_case], path=path)
    # run('Logistic Regression', test=test, target_test=target_test, train=train,target_train=target_train)
    
    # compare_models(models=models,train=train,val=val,test=test,target_train=target_train,target_val=target_val,target_test=target_test)