from django.urls import path
from .views import process_csv
from .views import supp_col

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('supp_col/', supp_col, name='supp_col'),
]
