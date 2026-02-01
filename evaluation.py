from sklearn.metrics import classification_report
from training import pred_with_threshold

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

def evaluate_test(pred_test, target_test, model,train,val,target_train,target_val):
    pred = pred_with_threshold(model=model, train=train,val=val,test=pred_test,target_train=target_train,target_val=target_val,target_test=target_test)
    print(pred.optimal_threshold)
    return (pred.pred_test_opt >= pred.optimal_threshold)