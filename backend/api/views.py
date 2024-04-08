from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
import json

def process_csv(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    
    num_rows, num_columns = df.shape
    
    imputation_method = request.POST.get('imputation_method','mode')
    
    missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()      
    total_missing_percentage = sum(missing_percentage.values()) 
    
    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)
    df_imputed = impute_missing_values(df, method=imputation_method)
    
    return JsonResponse({'dataset': json_obj,
                         #'imputed_dataset': df_imputed.to_dict(orient='records'),
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                          'num_rows': num_rows,
                         'num_columns': num_columns
                         })
    
def impute_missing_values(df, method='mode'):
    if method == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mean':
        return df.fillna(df.mean())
    
#Suppression des colonnes des missing values 
    
def supp_col(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    df_supp =  df.dropna() 
    df_supp = df_supp.to_json(orient='records')
    json_obj = json.loads(df_supp)
    
    return JsonResponse({'dataset': json_obj
                         })
    