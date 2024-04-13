from django.shortcuts import render
import pandas as pd
import seaborn as sns
import base64
import io
import base64
import matplotlib.pyplot as plt
from django.http import JsonResponse
import plotly.express as px
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
        
#Suppression des lignes des missing values 
def delete_missing_row(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            dataset = body_data.get('dataset', [])
            df = pd.DataFrame(dataset)
            df_supp = df.dropna() 
            json_str = df_supp.to_json(orient='records')
            json_obj = json.loads(json_str)
            
            num_rows, num_columns = df_supp.shape    
            missing_percentage = (df_supp.isnull().sum() / df_supp.shape[0] * 100).to_dict()  
            print(missing_percentage)    
            total_missing_percentage = sum(missing_percentage.values()) 
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                          'num_rows': num_rows,
                         'num_columns': num_columns
                         })
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
    
#linear_regression    
def linear_regression(request):
     if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        dataset = body_data.get('dataset', [])
        selected_columns = body_data.get('selected_columns', [])
        target = body_data.get('target', '')

        df = pd.DataFrame(dataset)

        plot = sns.pairplot(df, x_vars=selected_columns, y_vars=target, height=7, aspect=0.7,
                            kind='reg', plot_kws={'ci': None, 'line_kws': {'color': 'red'}})

        buffer = io.BytesIO()
        plot.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_img = base64.b64encode(buffer.read()).decode('utf-8')

        return JsonResponse({'encoded_img': encoded_img})

     else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
        
def save_state(request):
    
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        selected_columns = body_data.get('selected_columns', [])
        
        # Enregistrement des colonnes sélectionnées dans les cookies
        response = JsonResponse({'message': 'État enregistré avec succès'})
        response.set_cookie('selected_columns', json.dumps(selected_columns))
        
        return response
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def load_state(request):
     if request.method == 'GET':
         
         # Récupération des colonnes sélectionnées à partir des cookies
         selected_columns = json.loads(request.COOKIES.get('selected_columns', '[]'))
        
         return JsonResponse({'selected_columns': selected_columns})
     else:
         return JsonResponse({'error': 'Method not allowed'}, status=405)


def generate_plot_data(request):
    df = px.data.tips()
    fig = px.scatter(
        df, x='total_bill', y='tip', opacity=0.65,
        trendline='ols', trendline_color_override='darkblue'
    )

    # Convert the Plotly figure to JSON
    plot_data = fig.to_json()

    return JsonResponse({'plot_data': plot_data})