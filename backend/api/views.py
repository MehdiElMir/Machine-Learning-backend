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
""" def supp_col(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    df_supp =  df.dropna() 
    df_supp = df_supp.to_json(orient='records')
    json_obj = json.loads(df_supp)
    
    return JsonResponse({'dataset': json_obj
                         }) """

def supp_col(request):
    # Check if the request is a POST request
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' in request.FILES:
            # If a file is uploaded, process it as before
            uploaded_file = request.FILES['file']
            df = pd.read_csv(uploaded_file)
            df_supp = df.dropna() 
            df_supp = df_supp.to_json(orient='records')
            json_obj = json.loads(df_supp)
        else:
            # If no file is uploaded, process raw JSON data
            try:
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                # Assuming the raw JSON contains 'dataset' key
                dataset = body_data.get('dataset', [])
                df = pd.DataFrame(dataset)
                df_supp = df.dropna() 
                json_obj = df_supp.to_dict(orient='records')
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'dataset': json_obj})
    else:
        # Return error response for methods other than POST
        return JsonResponse({'error': 'Method not allowed'}, status=405)
                         
    
#Suppression des colonnes
""" 
def supp_col_checked(request):  
      
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    
    col = request.POST.get('col', '')
    
    df = df.drop(columns=[col])
    df_supp = df.to_json(orient='records')
    json_obj = json.loads(df_supp)
   
    return JsonResponse({'dataset': json_obj
                         }) """
                         
def supp_col_checked(request):
    if request.method == 'POST':
        try:
            if 'file' in request.FILES:
                uploaded_file = request.FILES['file']
                df = pd.read_csv(uploaded_file)
            else:
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                dataset = body_data.get('dataset', [])
                col = body_data.get('col', '')
                if not dataset:
                    return JsonResponse({'error': 'Empty dataset'}, status=400)
                if not col:
                    return JsonResponse({'error': 'Column not specified'}, status=400)
                df = pd.DataFrame(dataset)
                if col not in df.columns:
                    return JsonResponse({'error': f'Column "{col}" not found in dataset'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
        
        try:
            df = df.drop(columns=[col])
            df_supp = df.to_json(orient='records')
            json_obj = json.loads(df_supp)
            return JsonResponse({'dataset': json_obj})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
