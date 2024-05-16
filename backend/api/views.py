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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.tree import DecisionTreeRegressor  
from sklearn.preprocessing import LabelEncoder

def process_csv(request):
    uploaded_file = request.FILES['file']
    df = pd.read_csv(uploaded_file)
    num_rows, num_columns = df.shape    
    
    missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_dict()      
    total_missing_percentage = sum([True for idx,row in df.iterrows() if any(row.isnull())]) 
    numeric_columns_names = df.select_dtypes(include='number').columns.tolist() 
    categorical_columns_names = df.select_dtypes(include=['object', 'category']).columns.tolist() 

    
    #mean_values = df.mean().to_dict()
    
    json_str = df.to_json(orient='records')
    json_obj = json.loads(json_str)
    
    
    return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                         'num_rows': num_rows,
                         #'mean_values': mean_values,
                         'num_columns': num_columns,
                         'numeric_columns_names':numeric_columns_names,
                         'categorical_columns_names':categorical_columns_names
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
            categorical_columns_names = df.select_dtypes(include=['object', 'category']).columns.tolist()
 
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'dataset': json_obj,
                         'missing_percentage': missing_percentage,
                         'total_missing_percentage': total_missing_percentage,
                          'num_rows': num_rows,
                         'num_columns': num_columns,
                         'numeric_columns_names':numeric_columns_names,
                         'categorical_columns_names':categorical_columns_names
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
        categorical_columns_names = df.select_dtypes(include=['object', 'category']).columns.tolist()
 
        
        json_str = df.to_json(orient='records')
        json_obj = json.loads(json_str)
        return JsonResponse({'dataset': json_obj,
                            'missing_percentage': missing_percentage,
                            'total_missing_percentage': total_missing_percentage,
                            'num_rows': num_rows,
                            'num_columns': num_columns,
                            'numeric_columns_names':numeric_columns_names,
                            'categorical_columns_names':categorical_columns_names
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
            categorical_columns_names = df.select_dtypes(include=['object', 'category']).columns.tolist()
 
            
            json_str = df.to_json(orient='records')
            json_obj = json.loads(json_str)
            return JsonResponse({'dataset': json_obj,
                                'missing_percentage': missing_percentage,
                                'total_missing_percentage': total_missing_percentage,
                                'num_rows': num_rows,
                                'num_columns': num_columns,
                                'numeric_columns_names':numeric_columns_names,
                                'categorical_columns_names':categorical_columns_names
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
        num_rows, num_columns = balanced_df.shape

        json_str = balanced_df.to_json(orient='records')
        json_obj = json.loads(json_str)


        return JsonResponse({'data': json_obj,
                             'num_rows': num_rows,
                             })
    
def knn_classification(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            dataset = body_data.get('dataset', [])
            target = body_data.get('target', '')
            n_neighbors = int(body_data.get('n_neighbors', 5))  # Default to 5 neighbors if not provided

            df = pd.DataFrame(dataset)

            # Extract features (X) and target (y) from the DataFrame
            X = df.drop(columns=target)
            # Convert categorical features to numeric using one-hot encoding
            X = pd.get_dummies(X)

            # Encode the target column
            le = LabelEncoder()
            y = le.fit_transform(df[target])

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=0
            )

            # Initialize and train the KNN classifier
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)

            # Predict probabilities on the test set
            y_score = clf.predict_proba(X_test)[:, 1]

            # Create DataFrame for the plot with test data and predictions
            df_plot = pd.DataFrame({
                'x': X_test.iloc[:, 0],
                'y': X_test.iloc[:, 1],
                'true_label': le.inverse_transform(y_test),  # Use original category names
                'score': y_score
            })

            # Create the scatter plot with Plotly Express
            fig = px.scatter(
                df_plot, x='x', y='y',
                color='score', color_continuous_scale='RdBu',
                symbol='true_label',
                labels={'symbol': 'label', 'color': 'score of <br>first class'}
            )
            fig.update_traces(marker_size=12, marker_line_width=1.5)
            fig.update_layout(legend_orientation='h')

            # Convert the plot to JSON
            plot_data = fig.to_json()

            # Return the plot data in the JSON response
            return JsonResponse({'plot_data': json.loads(plot_data)})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

def knn_regression(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            # Extraire les données et paramètres du corps de la requête JSON
            dataset = body_data.get('dataset', [])
            target = body_data.get('target', '')
            n_neighbors = body_data.get('n_neighbors', '')  # Nombre de voisins par défaut

            # Créer le DataFrame à partir du dataset
            df = pd.DataFrame(dataset)
            
            # Vérifier la validité des données
            if not df.empty and target in df.columns:
                # Séparer les caractéristiques (X) et la cible (y)
                X = df.drop(columns=[target])
                y = df[target].astype(float)  # Assurez-vous que la cible est de type float

                # Division des données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=0
                )

                # Initialiser et entraîner le modèle de régression KNN
                clf = KNeighborsRegressor(n_neighbors=n_neighbors)
                clf.fit(X_train, y_train)

                # Prédire les valeurs sur l'ensemble de test
                y_pred = clf.predict(X_test)

                # Récupérer les noms des features (colonnes de X_test)
                feature_names = X_test.columns.tolist()

                # Créer un DataFrame avec les données de test et les prédictions
                df_pred = pd.DataFrame({
                    'feature_1': X_test.iloc[:, 0],
                    'feature_2': X_test.iloc[:, 1],
                    'true_target': y_test,
                    'predicted_target': y_pred
                })

                # Créer un scatter plot avec Plotly Express en utilisant les noms des features
                fig = px.scatter(
                    df_pred, x='feature_1', y='feature_2',
                    color='predicted_target', color_continuous_scale='RdBu',
                    labels={'color': 'predicted target'},
                    title='KNN Regression Prediction'
                )

                fig.update_traces(marker_size=12, marker_line_width=1.5)
                fig.update_layout(legend_orientation='h')

                # Renommer les axes x et y avec les noms des features
                fig.update_xaxes(title_text=feature_names[0])
                fig.update_yaxes(title_text=feature_names[1])

                # Conversion du graphique en format JSON
                plot_data = fig.to_json()
                json_obj = json.loads(plot_data)

                # Retourner les données du graphique dans la réponse JSON
                return JsonResponse({'plot_data': json_obj})

            else:
                return JsonResponse({'error': 'Le dataset est vide ou la cible spécifiée est incorrecte.'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)
    


def undersampling(request):
     if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        dataset = body_data.get('dataset', [])
        target = body_data.get('target','')
        df = pd.DataFrame(dataset)

        X = df.drop([target], axis=1)
        y = df[target]
        
        #rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
        rus = RandomUnderSampler(sampling_strategy="not minority") # String
        X_res, y_res = rus.fit_resample(X, y)

        y_res_list= y_res.tolist()

        balanced_df = pd.DataFrame(X_res, columns=X.columns)

        balanced_df = pd.concat([balanced_df, pd.Series(y_res_list, name = target)], axis=1)

        num_rows, num_columns = balanced_df.shape

        json_str = balanced_df.to_json(orient='records')
        json_obj = json.loads(json_str)


        return JsonResponse({'data': json_obj,'num_rows': num_rows})  

def generate_value_counts(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)

            dataset = body_data.get('dataset', [])
            target = body_data.get('target', '')
            df = pd.DataFrame(dataset)

            # Count occurrences of each category in the target column
            value_counts = df[target].value_counts()

            # Format the counts to the desired structure
            formatted_counts = [{'value': count, 'name': name} for name, count in value_counts.items()]

            # Return the formatted value counts in the JSON response
            return JsonResponse({'values': formatted_counts})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    
def decision_tree(request):
     if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        dataset = body_data.get('dataset', [])
        target = body_data.get('target','')
        column = body_data.get('column','')

        df = pd.DataFrame(dataset)

        X = df[column].values[:, None]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, df[target], random_state=42)

        model = DecisionTreeRegressor()
        
        model.fit(X_train, y_train)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        

        fig = go.Figure([
            go.Scatter(x=X_train.squeeze(), y=y_train, 
                    name='train', mode='markers'),
            go.Scatter(x=X_test.squeeze(), y=y_test, 
                    name='test', mode='markers'),
            go.Scatter(x=x_range, y=y_range, 
                    name='prediction')
        ])
        
        plot_data = fig.to_json()
        json_obj = json.loads(plot_data)

        return JsonResponse({'plot_data': json_obj})    
    
    
    
    
       
    
    
    
    
    
    