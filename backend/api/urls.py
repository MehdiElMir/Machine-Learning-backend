# your_app/urls.py
from django.urls import path
from .views import process_csv

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
]
