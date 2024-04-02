from django.shortcuts import render

# Create your views here.

import pandas as pd
from django.http import JsonResponse

def process_csv(request):
    # Assuming 'file' is the key for the uploaded CSV file in the request
    uploaded_file = request.FILES['file']
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    number = 20
    # Perform data analysis on df (e.g., machine learning techniques)
    # Example:
    #processed_data = df.mean()  # Just an example, replace with your actual processing
    
    # Convert DataFrame to a list of dictionaries
    processed_data_json = df.reset_index().to_dict(orient='records')
    
    # Return processed data as JSON response
    return JsonResponse({'processed_data': processed_data_json,
                         'missing_values': number})
