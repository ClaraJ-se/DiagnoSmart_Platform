import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from utils import get_path
import seaborn as sns

from math import sqrt
import os, random
from sklearn.metrics import roc_auc_score
from  matplotlib import pyplot as plt
import xgboost as xgb
import sklearn.metrics as skmet
from tqdm import tqdm
from typing import List
from sklearn.model_selection import ParameterGrid

def evaluation(model, selected_features, x_test, y_test, paths,name='test'):
    x_test_selected = x_test[selected_features]
    prob_test = model.predict_proba(x_test_selected)
    lossvalue = 0
    
    print('shapes',np.shape(prob_test),np.shape(y_test))
    fpr, tpr, thresholds = skmet.roc_curve(y_test,prob_test[:,1])
    auprc = skmet.average_precision_score(y_test,prob_test[:,1])
    auc = skmet.roc_auc_score(y_test,prob_test[:,1])
    
    fig=plt.figure()
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve val test')
    plt.ylabel('Sensitivity (TPR)')
    plt.xlabel('1-Specificity (FPR)')
    plt.savefig(os.path.join(paths['training'],f'ROC_{name}.png'), bbox_inches='tight', dpi=1200)
        
    
    gmean_inter,thresh_inter=[], []
    
    for threshold in thresholds:
        TN, FP, FN, TP = skmet.confusion_matrix(y_test, prob_test[:,1] >= threshold).ravel()
        spec=TN/(TN+FP)
        sens=TP/(TP+FN)
        gmean_inter.append(sqrt(spec*sens))
        thresh_inter.append(threshold)
        
    bestT=thresh_inter[np.where(gmean_inter==np.max(gmean_inter))[0][0]]

    TP, FP, TN, FN = perf_measure(y_test, np.where(prob_test[:,1]>=bestT,1,0))                      
    spec = TN/(TN+FP)
    sens = TP/(TP+FN)
    prec = TP/(TP+FP)
    acc = (TP+TN)/(TP+TN+FP+FN)
    final_results_columns = np.concatenate([['specificity','sensitivity','gmean', 
                                            'auprc','auc','accuracy', 'precision',
                                            'F_measure','threshold','lossValid']])

    final_test = [np.concatenate([[spec,sens,sqrt(spec*sens),auprc,auc,acc,prec,
                                2*prec*sens/(prec+sens),bestT,lossvalue]])]
    
    final_test = pd.DataFrame(final_test, columns =final_results_columns)
    final_test.to_csv(os.path.join(paths['training'],f'{name}_results.csv '))

    return final_test, x_test_selected

class List_CV_metrics:
    def __init__(self):
        self.spe = []
        self.sens = []
        self.gmean = []
        self.auprc = []
        self.auc = []
        self.accuracy = []
        self.precision = []
        self.fmeatrics = []
        self.threshold_list = []
        self.loss = []
        
    def update_values(self,specificity : float,sensitivity : float,precision : float, auc : float, auprc: float, accuracy: float, loss_value : float, best_threshold : float):
        self.spe.append(specificity)
        self.sens.append(sensitivity)
        self.gmean.append(sqrt(specificity*sensitivity))
        self.auprc.append(auprc)
        self.auc.append(auc)
        self.accuracy.append(accuracy)
        self.precision.append(precision)
        if precision != 0:
            self.fmeatrics.append(2*precision*sensitivity/(precision+sensitivity))
        else:
            self.fmeatrics.append(0)
        self.threshold_list.append(best_threshold)
        self.loss.append(loss_value)

def perf_measure(y_actual: list, y_hat : list):
    """calculate confusion matrix

    Args:
        y_actual (list): list of ground truth
        y_hat (list): list of predictions
    """
    TP,FP,TN,FN = 0,0,0,0
    if len(np.shape(y_actual)) == 2:
        y_actual = np.array(y_actual)[:,0]
    else:
        y_actual = np.array(y_actual)
    for i in range(len(y_hat)):
        if y_actual[i]==1 and y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

def optimisation(parameters_to_test,size_grid,final_results,X_train,Y_train,X_val,Y_val,paths):
    grid = ParameterGrid(parameters_to_test)
    randomsearch = random.sample(range(0, len(grid)), size_grid)    
    results = [[] for i in range(5)]
    
    for params_index in randomsearch:
        params = grid[params_index]
        
        for k in params.keys():
            params[k] = [params[k]]
        param = pd.DataFrame.from_dict(params)
        
        current_param_list = param.iloc[0]
        
        for res in final_results:
            if np.array_equal(current_param_list, res[-len(current_param_list):]):
                print(current_param_list, 'already calculated')
                randomsearch.remove(params_index)
                continue

        
    best_metrics = 'gmean'

    for params_index in tqdm(randomsearch):
        params = grid[params_index]       
        param = params.copy()
        for k in param.keys():
            param[k] = [param[k]]
            
        param = pd.DataFrame.from_dict(param)
        
        remove_point= str.maketrans(".", '_') #to create a file name based on param we need to remove points
        name_of_combination = [str(param.iloc[0,i]).translate(remove_point) for i in range(len(param.iloc[0]))]
        
        
        final_results_columns = np.concatenate([['specificity','sensitivity','gmean', 
                                                 'auprc','auc','accuracy', 'precision',
                                                 'F_measure','threshold','lossValid'],
                                                param.columns])

        metrics_cv = List_CV_metrics()
        for CV in tqdm(range(0, 5), desc = 'CV'):
            path_cv = os.path.join(paths['training'],f'CV{CV}')
            os.makedirs(path_cv, exist_ok=True) 
            
            param_name = '' 
            for i in range(len(param.columns)):
                param_name = param_name + param.columns[i][0:3] + name_of_combination[i] + '_'
            
            #? select patients from the CV
            xtrain = X_train[CV]
            ytrain = Y_train[CV]
            xval = X_val[CV]
            yval = Y_val[CV]
            
            evalset = [(xtrain, ytrain), (xval,yval)]
            
            params_without_num_feat = params.copy()
            del params_without_num_feat['num_feat']
            
            #? create a model before feature selection
            model = xgb.XGBClassifier(**params_without_num_feat, 
                                          objective="binary:logistic", 
                                          early_stopping_rounds=5, 
                                          eval_metric=['auc','logloss'], 
                                          use_label_encoder=False,max_delta_step=1) 
                
            model.fit(xtrain,ytrain,eval_set=evalset, verbose=False) 
                
            #? feature selection
            selected_features = select_features(model,xtrain,param['num_feat'][0])
                   
            xtrain_selected = xtrain[selected_features]
            xval_selected = xval[selected_features]
                        
            evalset = [(xtrain_selected, ytrain), (xval_selected,yval)]
            
            #? train model after feature selection
            model = xgb.XGBClassifier(**params_without_num_feat, 
                                          objective="binary:logistic", 
                                          early_stopping_rounds=5, 
                                          eval_metric=['auc','logloss'], 
                                          use_label_encoder=False,
                                          max_delta_step=1) 
            model.fit(xtrain_selected,ytrain,eval_set=evalset, verbose=False) 
            
            results_evaluation = model.evals_result()
            loss_value = results_evaluation['validation_1']['logloss'][-1] 
                 
            #? evaluate the model             
            prob_val = model.predict_proba(xval_selected)
            
            if len(np.shape(prob_val))==2:
                prob_val = prob_val[:,1]
                
            fpr, tpr, thresholds = skmet.roc_curve(yval,prob_val)
            
            plot_figures_training(results_evaluation,path_cv,param_name,tpr,fpr)
            
            metrics_cv,results = calculate_training_metrics(CV,results, yval,prob_val, loss_value, thresholds,list(params.values()), metrics_cv, best_metrics)
        
        df_final_results = pd.DataFrame(final_results, columns = final_results_columns)
        
        if (np.mean(metrics_cv.precision)+np.mean(metrics_cv.sens)) != 0:
            f_metrics_mean = (np.mean(metrics_cv.precision)+np.mean(metrics_cv.sens))
        else:
            f_metrics_mean = 0
            
        final_results.append(np.concatenate([[np.mean(metrics_cv.spe), np.mean(metrics_cv.sens),
                                  sqrt(np.mean(metrics_cv.spe)*np.mean(metrics_cv.sens)),
                                  np.mean(metrics_cv.auprc),
                                  np.mean(metrics_cv.auc),np.mean(metrics_cv.accuracy),np.mean(metrics_cv.precision),
                                  f_metrics_mean,
                                  np.mean(metrics_cv.threshold_list),np.mean(metrics_cv.loss)],list(params.values())])   )
        
    final_results = pd.DataFrame(final_results, columns = final_results_columns)


    df_final_results = df_final_results.sort_values(best_metrics, ascending=False)
    df_final_results.to_csv(os.path.join(paths['training'],'optimisation_results.csv'))
        # hyperparameters selection

    hyperparams_best = df_final_results.sort_values(best_metrics, ascending=False).iloc[0,10:]
    best_val = final_results.sort_values(best_metrics, ascending=False).iloc[0,:10]
    
    return model, hyperparams_best, best_val, df_final_results,selected_features 
    

def get_stratification(data:pd.DataFrame):

    # Suppose df est votre DataFrame et vous voulez faire une stratification basée sur les colonnes 'col1' et 'col2'
    # Vous pouvez créer une nouvelle colonne 'combined' qui concatène les valeurs de 'col1' et 'col2'
    data['stratif'] = data['gender'] + '_' + data['infarctus'].astype(str)

    # Ensuite, vous pouvez utiliser StratifiedKFold avec la nouvelle colonne 'combined'
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    train_data = []
    val_data = []
    c=0
    for train_index, val_index in skf.split(data,  data['stratif']):
        if c != 5:
            train_data.append(data.iloc[train_index])
            val_data.append(data.iloc[val_index])
        else:
            test_data = data.iloc[val_index]
        c+=1

    print('Info on test data')
    get_info(test_data)

    for i in range(5):
        print(f"train info {i}")
        get_info(train_data[i])
    for i in range(5):
        print(f"val info {i}")
        get_info(val_data[i])

    return train_data, val_data, test_data


def get_info(df):
    
   df_men = df[df.gender == 'M']
   df_women = df[df.gender == 'F']
   df_infarctus_pos = df[df.infarctus == 1]
   df_infarctus_neg = df[df.infarctus == 0]
   df_infarctus_pos_women = df_infarctus_pos[df_infarctus_pos.gender == 'F']
   df_infarctus_neg_women = df_infarctus_neg[df_infarctus_neg.gender == 'F']
   df_infarctus_pos_men = df_infarctus_pos[df_infarctus_pos.gender == 'M']
   df_infarctus_neg_men = df_infarctus_neg[df_infarctus_neg.gender == 'M']

   
   print(f"Number Acute myocardial infarction: {df_infarctus_pos['stay_id'].nunique()}")
   print(f"Total number of stays: {df['stay_id'].nunique()}")
   print(f"Rate of Acute myocardial infarction: {df_infarctus_pos['stay_id'].nunique()/df['stay_id'].nunique()}")  

   
   print(f"Women: Number Acute myocardial infarction: {df_infarctus_pos_women['stay_id'].nunique()}")
   print(f"Women: Total number of stays: {df_women['stay_id'].nunique()}")
   print(f"Women: Rate of Acute myocardial infarction: {df_infarctus_pos_women['stay_id'].nunique()/df_women['stay_id'].nunique()}")  

   print(f"Men: Number Acute myocardial infarction: {df_infarctus_pos_men['stay_id'].nunique()}")
   print(f"Men: Total number of stays: {df_men['stay_id'].nunique()}")
   print(f"Men: Rate of Acute myocardial infarction: {df_infarctus_pos_men['stay_id'].nunique()/df_men['stay_id'].nunique()}")  

   print(f"Infarctus: Rate of women: {df_infarctus_pos_women['stay_id'].nunique()/df_infarctus_pos['stay_id'].nunique()}")   


def plot_figures_training(results_evaluation, pathToCV : str, param_name : str, TPR : int, FPR : int):
        
            
    # plot learning curves
    fig=plt.figure()
    plt.plot(results_evaluation['validation_0']['auc'], label='train')
    plt.plot(results_evaluation['validation_1']['auc'], label='validation')
    
    plt.title('AUC value over the iterations/number of trees')
    plt.ylabel('AUC')
    plt.xlabel('Iterations')
    plt.savefig(f'{pathToCV}/{param_name}_learningAUC.png', bbox_inches='tight', dpi=1200)
    plt.legend()
    #plt.show()
    plt.close(fig)
    
    fig=plt.figure()
    plt.plot(results_evaluation['validation_0']['logloss'], label='train')
    plt.plot(results_evaluation['validation_1']['logloss'], label='validation')
    plt.title('Loss value over the iterations/number of trees')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(os.path.join(pathToCV,f'{param_name}_learning.png'), bbox_inches='tight', dpi=1200)
    #plt.show()
    plt.close(fig)
    
    fig=plt.figure()
    plt.plot(FPR,TPR)
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve val')
    plt.ylabel('Sensitivity (TPR)')
    plt.xlabel('1-Specificity (FPR)')
    plt.savefig(os.path.join(pathToCV,f'{param_name}ROC_val.png'), bbox_inches='tight', dpi=1200)
    plt.close(fig)


def calculate_training_metrics(CV: int,results : List[list], yval: list,prob_val: list, 
                               loss_value : float, thresholds : list,parameters : list,
                               list_csv_metrics : List_CV_metrics, best_metrics: str = 'gmean'):
    """Calculate training metrics

    Args:
        CV (int): index of the current cross validation fold
        results (List[list]): list of results
        yval (list): ground truth values
        prob_val (list): predicted probabilities
        loss_value (float): loss values
        thresholds (list): list of thresholds
        parameters (list): list of hyperparameters
        list_csv_metrics (List_CV_metrics): metrics to calculate
        best_metrics (str, optional): metrics to select the best hyperparameters. Defaults to 'gmean'.

    Returns:
        list_csv_metrics
        results
    """
    gmean_inter,thresh_inter,auc_inter=[],[],[]
    
    if len(np.shape(prob_val))==2:
        prob_val = prob_val[:,1]
        
    for threshold in thresholds:
        TN, FP, FN, TP = skmet.confusion_matrix(yval, prob_val >= threshold).ravel()
        specificity=TN/(TN+FP)
        sensitivity=TP/(TP+FN)
        roc_auc = roc_auc_score(yval, prob_val >= threshold)
        gmean_inter.append(sqrt(specificity*sensitivity))
        auc_inter.append(roc_auc)
        thresh_inter.append(threshold)
        
    if best_metrics=='gmean':
        metrics_inter = gmean_inter
        
    elif best_metrics == 'auc':
        metrics_inter = auc_inter
        
    best_threshold=thresh_inter[np.where(metrics_inter==np.max(metrics_inter))[0][0]]  
    auprc = skmet.average_precision_score(yval,prob_val)
    auc = skmet.roc_auc_score(yval,prob_val)
    
    #tpr_80 = np.argwhere(tpr == find_nearest(tpr, 0.8))[0]
    #threshold = thresholds[tpr_80.item()]
    TP, FP, TN, FN = perf_measure(np.array(yval), prob_val>=best_threshold)
            
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    if precision+sensitivity != 0:
        f_metrics = 2*precision*sensitivity/(precision+sensitivity)
    else:
        f_metrics = 0
    results[CV].append(np.concatenate([[specificity,
                        sensitivity,
                        sqrt(specificity*sensitivity),
                        auprc,auc,accuracy,precision,
                        f_metrics,
                        best_threshold],parameters]))

    list_csv_metrics.update_values(specificity, sensitivity, precision, auc, auprc, accuracy, loss_value, best_threshold)
    
    return list_csv_metrics,results
    
    
def select_features(model,X,num_feat):
    imp=model.feature_importances_
    
    sorted_idx = imp.argsort()
    
    var_imp = [X.columns[sorted_idx],imp[sorted_idx]]
    
    index_first_feature = len(X.columns) - num_feat
    
    selected_features = var_imp[0][index_first_feature:]
    
    return selected_features

if __name__ == "__main__":

    paths = get_path()

    selection = pd.read_csv(os.path.join(paths['selection'],'patients.csv'))

    list_symptoms = [ 'pain','jaundice','hyperglycemia', 'dehydration', 'Hematemesis', 'distention', 'nausea',
       'swelling', 'tachycardia', 'bleed', 'fatigue', 'fever', 'cough', 'itch',
       'paralysis', 'diarrhea', 'dizzy', 'hemorroids', 'neurologic',
       'lump', 'numbness', 'seizure', 'migraine', 'sore', 'smelling urine',
       'hearing loss', 'rash_redness', 'hypoglycemia', 'dyspnea', 'anemia',
       'throat foreign body sensation', 'constipation', 'dysuria', 'anxiety',
       'hematuria', 'pain_back', 'pain_neck',
       'pain_chest', 'pain_joint', 'pain_abdominal', 'pain_head',
       'pain_urinary track', 'paralysis_face', 'paralysis_arm',
       'cramps_abdominal', 'pain_arm_left']

    list_vital = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

    list_params = list_vital + list_symptoms

    list_columns_param = ['stay_id'] + list_params 

    list_columns_gt = ['stay_id','infarctus']

    list_columns_info = list_columns_gt + [ 'hadm_id', 'intime', 
                                           'outtime', 'gender', 
                                           'race', 'arrival_transport',
                                             'disposition']
    
    data = selection[list_columns_gt + ['gender'] + list_params]


    #! remove feature where all 0
    zero_columns = data.columns[(data == 0).all()]
    list_symptoms = [symptom for symptom in list_symptoms if symptom not in zero_columns]
    list_symptoms = [symptom for symptom in list_symptoms if symptom not in ['pain']]
    # 37 symptoms left
    list_params = list_vital + list_symptoms
    # 43 features left

    list_columns_param = ['stay_id'] + list_params 

    data = data[list_columns_gt + ['gender'] + list_params]

    #! study correlation with variable of interest

    data_corr = data[['infarctus','gender'] + list_params]
    correlation_matrix = data_corr.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlation_values = abs(upper_triangle.stack()).sort_values(ascending=False )
    top_10_correlated = correlation_values.head(10)
    print("Top 10 most correlated parameters:")
    print(top_10_correlated)

    data_corr_women = data_corr[data.gender == 'F']
    data_corr_men= data_corr[data_corr.gender == 'M']


    correlation_matrix_men = data_corr_men.corr()
    upper_triangle_men  = correlation_matrix_men.where(np.triu(np.ones(correlation_matrix_men.shape), k=1).astype(bool))
    correlation_values_men  = abs(upper_triangle_men.stack()).sort_values(ascending=False )
    infarctus_corr_men = correlation_matrix_men['infarctus'].abs().sort_values(ascending=False)
    top_10_correlated_men  = infarctus_corr_men.head(11)
    print("Top 10 most correlated parameters for men:")
    print(top_10_correlated_men )

    correlation_matrix_women  = data_corr_women.corr()
    upper_triangle_women = correlation_matrix_women.where(np.triu(np.ones(correlation_matrix_women.shape), k=1).astype(bool))
    correlation_values_women = abs(upper_triangle_women.stack()).sort_values(ascending=False )
    infarctus_corr_women = correlation_matrix_women['infarctus'].abs().sort_values(ascending=False)
    top_10_correlated_women  = infarctus_corr_women.head(11)
    print("Top 10 most correlated parameters for women:")
    print(top_10_correlated_women)

    #! double stratification 
    print('stratification')
    train_data, val_data, test_data = get_stratification(data)

    x_test = test_data[list_params] # 304 stays
    y_test = test_data['infarctus']

    X_train = [train_data[i][list_params] for i in range(len(train_data))]    
    Y_train = [train_data[i]['infarctus'] for i in range(len(train_data))]
    X_val = [val_data[i][list_params] for i in range(len(val_data))]    
    Y_val = [val_data[i]['infarctus'] for i in range(len(val_data))]

    
    print('optimisation')

    final_results = []
    
    parameters_to_test = {
        "eta"    : [0.03,0.05,0.1,0.08] ,
        "max_depth"        : [ 4,5,7,8,9,10,15],
        "min_child_weight" : [ 5,10,15,20,25,30,50,100],
        "gamma"            : [  1,5,10 ],
        "colsample_bytree" : [ 0.7,0.8],
        "n_estimators"     : list(range(30,40)),
        "subsample" : [0.8,0.5,0.6],
        "num_feat":list(range(5,len(x_test.columns))),
        "learning_rate": [0.03,0.05,0.06,0.07,0.08,0.1]
        }
    
    size_grid = 2

    model, hyperparams_best, best_val, df_final_results, selected_features = optimisation(parameters_to_test,size_grid,final_results,X_train,Y_train,X_val,Y_val,paths)

    # subsample            0.50
    # num_feat            18.00
    # n_estimators        32.00
    # min_child_weight    15.00
    # max_depth           15.00
    # learning_rate        0.03
    # gamma               10.00
    # eta                  0.10
    # colsample_bytree     0.70

    final_test, x_test_selected = evaluation(model, selected_features, x_test, y_test, paths)


    final_test, x_test_selected = evaluation(model, selected_features, x_test, y_test, paths)

    test_data_women = test_data[test_data.gender == 'F']
    test_data_men = test_data[test_data.gender == 'M']
        
    x_test_women = test_data_women[list_params] # 304 stays
    y_test_women = test_data_women['infarctus']

    x_test_men = test_data_men[list_params] # 304 stays
    y_test_men = test_data_men['infarctus']

    final_test_women, x_test_women_selected = evaluation(model, selected_features, x_test_women, y_test_women, paths)
    final_test_men, x_test_men_selected = evaluation(model, selected_features, x_test_men, y_test_men, paths)


#! most important features
imp=model.feature_importances_
sorted_idx = imp.argsort()
var_imp = [x_test.columns[sorted_idx],imp[sorted_idx]]


# Create bar plot
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(var_imp)), var_imp, align='center')
plt.yticks(np.arange(len(var_imp)), var_imp)
plt.xlabel('Feature Importance')
plt.title('Top Most Important Features')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
plt.show()

