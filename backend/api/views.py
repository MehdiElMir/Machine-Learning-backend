from django.shortcuts import render
import pandas as pd
import seaborn as sns
import base64
import io
import base64
import matplotlib.pyplot as plt
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.svm import SVR
from sklearn.datasets import make_moons
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
#pip install -U kaleido  
def linear_regression(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            dataset = body_data.get('dataset', [])
            selected_columns = body_data.get('selected_columns', [])
            target = body_data.get('target', '')
            df = pd.DataFrame(dataset)
            fig = px.scatter(df, x=selected_columns[0], y=target, opacity=0.65,
                             trendline='ols', trendline_color_override='darkblue')
            
            buffer = io.BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return JsonResponse({'encoded_img': encoded_img})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)
        

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

def smote(request):
     if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        dataset = body_data.get('dataset', [])
        target = body_data.get('target','')
        df = pd.DataFrame(dataset)

        X = df.drop([target], axis=1)
        y = df[target]

        #ros = RandomOverSampler(sampling_strategy=1) 
        ros = RandomOverSampler(sampling_strategy="not majority")
        X_res, y_res = ros.fit_resample(X, y)


        y_res_list= y_res.tolist()

        balanced_df = pd.DataFrame(X_res, columns=X.columns)

        balanced_df = pd.concat([balanced_df, pd.Series(y_res_list, name = target)], axis=1)

        json_str = balanced_df.to_json(orient='records')
        json_obj = json.loads(json_str)


        return JsonResponse({'data': json_obj})
    
def knn_classification(request):
    if request.method == 'POST':
        try:
            X, y = make_moons(noise=0.3, random_state=0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.astype(str), test_size=0.25, random_state=0)

            clf = KNeighborsClassifier(15)
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)[:, 1]

            fig = px.scatter(
                X_test, x=0, y=1,
                color=y_score, color_continuous_scale='RdBu',
                symbol=y_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
                labels={'symbol': 'label', 'color': 'score of <br>first class'}
            )
            fig.update_traces(marker_size=12, marker_line_width=1.5)
            fig.update_layout(legend_orientation='h')

            buffer = io.BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return JsonResponse({'encoded_img': encoded_img})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)


def knn_regression(request):
    if request.method == 'POST':
        try:
            # Générer les données
            X, y = make_moons(noise=0.3, random_state=0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=0)

            # Créer et entraîner le modèle de régression KNN
            clf = KNeighborsRegressor(n_neighbors=15)
            clf.fit(X_train, y_train)

            # Prédire les valeurs sur les données de test
            y_pred = clf.predict(X_test)

            # Créer un DataFrame avec les données de test et les prédictions
            df_pred = pd.DataFrame(np.column_stack((X_test[:, 0], X_test[:, 1], y_test, y_pred)),
                                   columns=['feature_1', 'feature_2', 'true_target', 'predicted_target'])

            # Créer un scatter plot avec Plotly Express
            fig = px.scatter(df_pred, x='feature_1', y='feature_2', color='predicted_target',
                             color_continuous_scale='RdBu', labels={'color': 'predicted target'})

            fig.update_traces(marker_size=12, marker_line_width=1.5)
            fig.update_layout(legend_orientation='h')

            # Enregistrer le graphique dans un buffer BytesIO au format PNG
            buffer = io.BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)

            # Encoder l'image en base64
            encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Retourner l'image encodée dans la réponse JSON
            return JsonResponse({'encoded_img': encoded_img})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)