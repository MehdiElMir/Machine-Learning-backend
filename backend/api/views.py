from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
import json

def process_csv(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    num_rows, num_columns = df.shape    
    
    missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()      
    total_missing_percentage = sum(missing_percentage.values()) 
    
    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)
    
    return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                          'num_rows': num_rows,
                         'num_columns': num_columns
                         })
        
#Suppression des colonnes des missing values 
def delete_missing_row(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            dataset = body_data.get('dataset', [])
            df = pd.DataFrame(dataset)
            df_supp = df.dropna() 
            json_obj = df_supp.to_dict(orient='records')
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'dataset': json_obj})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
                         
    
#Suppression des colonnes
def delete_selected_columns(request):
    if request.method == 'POST':

        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        dataset = body_data.get('dataset', [])
        columns_to_delete = body_data.get('columns_to_delete', [])

        df = pd.DataFrame(dataset)

        df.drop(columns=columns_to_delete, inplace=True, errors='ignore')
        columns = list(df.columns)
        columns=json.dumps(columns)

        json_str = df.to_json(orient='records')
        json_obj = json.loads(json_str)
        return JsonResponse({'dataset': json_obj,
                             'columns':columns})
    else:
       
        return JsonResponse({'error': 'Method not allowed'}, status=405)

#imputation    
def imputate_selected_column(request):
    if request.method == 'POST':
        
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        dataset = body_data.get('dataset', [])
        selected_columns = body_data.get('selected_columns', [])
        option = body_data.get('option','')    
        df = pd.DataFrame(dataset)
        
        if option == 'mode':
            imputated_col = df[selected_columns].fillna(df[selected_columns].mode().iloc[0])
            df[selected_columns] = imputated_col
            
        elif option == 'median':
            imputated_col = df[selected_columns].fillna(df[selected_columns].median())
            df[selected_columns] = imputated_col

        elif option == 'mean':
            imputated_col = df[selected_columns].fillna(df[selected_columns].mean())
            df[selected_columns] = imputated_col
    
        json_str = df.to_json(orient='records')
        json_obj = json.loads(json_str)
        return JsonResponse({'dataset': json_obj})
    
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    


