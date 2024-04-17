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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

def process_csv(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    num_rows, num_columns = df.shape    
    
    missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()      
    total_missing_percentage = sum([True for idx,row in df.iterrows() if any(row.isnull())]) 
    numeric_columns_names = df.select_dtypes(include='number').columns.tolist() 
    
    #mean_values = df.mean().to_dict()
    
    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)
    
    return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                         'num_rows': num_rows,
                         #'mean_values': mean_values,
                         'num_columns': num_columns,
                         'numeric_columns_names':numeric_columns_names
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
            total_missing_percentage = sum([True for idx,row in df_supp.iterrows() if any(row.isnull())]) 
            numeric_columns_names = df_supp.select_dtypes(include='number').columns.tolist() 
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                          'num_rows': num_rows,
                         'num_columns': num_columns,
                         'numeric_columns_names':numeric_columns_names
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
        num_rows, num_columns = df.shape    
        missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()  
        total_missing_percentage = sum([True for idx,row in df.iterrows() if any(row.isnull())]) 
        numeric_columns_names = df.select_dtypes(include='number').columns.tolist() 
        
        json_str = df.to_json(orient='records')
        json_obj = json.loads(json_str)
        return JsonResponse({'dataset': json_obj,
                            'missing_percentage': missing_percentage,
                            'total_missing_percentage': total_missing_percentage,
                            'num_rows': num_rows,
                            'num_columns': num_columns,
                            'numeric_columns_names':numeric_columns_names
                            })
    else:
       
        return JsonResponse({'error': 'Method not allowed'}, status=405)

#imputation    
def imputate_selected_column(request):
    if request.method == 'POST':
        
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            
            dataset = body_data.get('dataset', [])
            selected_columns = body_data.get('selected_columns', [])
            option = body_data.get('option','')    
            df = pd.DataFrame(dataset)
            
            if option == 'Mode':
                imputated_col = df[selected_columns].fillna(df[selected_columns].mode().iloc[0])
                df[selected_columns] = imputated_col
                
            elif option == 'Median':
                imputated_col = df[selected_columns].fillna(df[selected_columns].median())
                df[selected_columns] = imputated_col

            elif option == 'Mean':
                imputated_col = df[selected_columns].fillna(df[selected_columns].mean())
                df[selected_columns] = imputated_col
        
            json_str = df.to_json(orient='records')
            json_obj = json.loads(json_str)
            num_rows, num_columns = df.shape    
            missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()  
            total_missing_percentage = sum([True for idx,row in df.iterrows() if any(row.isnull())])
            numeric_columns_names = df.select_dtypes(include='number').columns.tolist() 
            
            json_str = df.to_json(orient='records')
            json_obj = json.loads(json_str)
            return JsonResponse({'dataset': json_obj,
                                'missing_percentage': missing_percentage,
                                'total_missing_percentage': total_missing_percentage,
                                'num_rows': num_rows,
                                'num_columns': num_columns,
                                'numeric_columns_names':numeric_columns_names
                                })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
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
    if request.method == 'GET':
        df = px.data.tips()
        fig = px.scatter(
            df, x='total_bill', y='tip', opacity=0.65,
            trendline='ols', trendline_color_override='darkblue'
        )

        # Convert the Plotly figure to JSON
        plot_data = fig.to_json()
        json_obj = json.loads(plot_data)

    return JsonResponse({'plot_data': json_obj})

def regression_linear_sckitlearn(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        dataset = body_data.get('dataset', [])
        selected_x = body_data.get('selected_x','')    
        selected_y = body_data.get('selected_y','') 
        
        df = pd.DataFrame(dataset)
          
        X = df[selected_x].values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, df[selected_y])

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
         
        fig = px.scatter(df, x = selected_x, y = selected_y, opacity=0.65)
        fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
        
        # Convert the Plotly figure to JSON
        plot_data = fig.to_json()
        json_obj = json.loads(plot_data)     

    return JsonResponse({'plot_data': json_obj})


def regression_linear_3D(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        dataset = body_data.get('dataset', [])
        selected_y= body_data.get('selected_y','')
        selected_x1= body_data.get('selected_x1','')
        selected_x2= body_data.get('selected_x2','') 
        df = pd.DataFrame(dataset)
   
        mesh_size = .02
        margin = 0
            
        X = df[[selected_x1, selected_x2]]
        y = df[selected_y]

        # Condition the model on sepal width and length, predict the petal width
        model = SVR(C=1.)
        model.fit(X, y)

        # Create a mesh grid on which we will run our model
        x_min, x_max = X[selected_x1].min() - margin, X[selected_x1].max() + margin
        y_min, y_max = X[selected_x2].min() - margin, X[selected_x2].max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        # Generate the plot
        fig = px.scatter_3d(df, x=selected_x1, y=selected_x2, z=selected_y)
        fig.update_traces(marker=dict(size=5))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
    
        plot_data = fig.to_json()
        json_obj = json.loads(plot_data) 
                   

    return JsonResponse({'plot_data': json_obj})

# def cross_validation(request):
#     if request.method == 'POST':
#         body_unicode = request.body.decode('utf-8')
#         body_data = json.loads(body_unicode)
        
#         dataset = body_data.get('dataset', [])
#         features = body_data.get('features', [])
#         target = body_data.get('target','')
#         df = pd.DataFrame(dataset)
        
    
#         N_FOLD = 6

#         # preprocess the data
#         X = df.drop(columns=features)
#         categorical_features =df.select_dtypes(exclude=["number","bool_","object_"])
#         X = pd.get_dummies(X, categorical_features)
#         y = df[target]

#         # Normalize the data
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Train model to predict life expectancy
#         model = LassoCV(cv=N_FOLD)
#         model.fit(X_scaled, y)
#         mean_alphas = model.mse_path_.mean(axis=-1)

#         fig = go.Figure([
#             go.Scatter(
#                 x=model.alphas_, y=model.mse_path_[:, i],
#                 name=f"Fold: {i+1}", opacity=.5, line=dict(dash='dash'),
#                 hovertemplate="alpha: %{x} <br>MSE: %{y}"
#             )
#             for i in range(N_FOLD)
#         ])
#         fig.add_traces(go.Scatter(
#             x=model.alphas_, y=mean_alphas,
#             name='Mean', line=dict(color='black', width=3),
#             hovertemplate="alpha: %{x} <br>MSE: %{y}",
#         ))

#         fig.add_shape(
#             type="line", line=dict(dash='dash'),
#             x0=model.alpha_, y0=0,
#             x1=model.alpha_, y1=1,
#             yref='paper'
#         )

#         fig.update_layout(
#             xaxis_title='alpha',
#             xaxis_type="log",
#             yaxis_title="Mean Square Error (MSE)"
#         )
        
#         plot_data = fig.to_json()
#         json_obj = json.loads(plot_data) 
                   

#     return JsonResponse({'plot_data': json_obj})


def cross_validation(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        dataset = body_data.get('dataset', [])
        features = body_data.get('features', [])
        target = body_data.get('target','')
        df = pd.DataFrame(dataset)
        
        N_FOLD = 6

        # Preprocess the data
        X = df.drop(columns=features)
        categorical_features = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(X, columns=categorical_features)
        y = df[target]

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model to predict target variable
        model = LassoCV(cv=N_FOLD)
        model.fit(X_scaled, y)
        mean_alphas = model.mse_path_.mean(axis=-1)

        # Create Plotly figure
        fig = go.Figure([
            go.Scatter(
                x=model.alphas_, y=model.mse_path_[:, i],
                name=f"Fold: {i+1}", opacity=.5, line=dict(dash='dash'),
                hovertemplate="alpha: %{x} <br>MSE: %{y}"
            )
            for i in range(N_FOLD)
        ])
        fig.add_traces(go.Scatter(
            x=model.alphas_, y=mean_alphas,
            name='Mean', line=dict(color='black', width=3),
            hovertemplate="alpha: %{x} <br>MSE: %{y}",
        ))

        fig.add_shape(
            type="line", line=dict(dash='dash'),
            x0=model.alpha_, y0=0,
            x1=model.alpha_, y1=1,
            yref='paper'
        )

        fig.update_layout(
            xaxis_title='alpha',
            xaxis_type="log",
            yaxis_title="Mean Square Error (MSE)"
        )
        
        plot_data = fig.to_json()
        json_obj = json.loads(plot_data) 

        return JsonResponse({'plot_data': json_obj})

    return JsonResponse({'error': 'Only POST requests are supported.'})


