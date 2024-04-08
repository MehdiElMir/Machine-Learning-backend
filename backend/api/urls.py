from django.urls import path
from .views import process_csv
from .views import supp_col
from .views import supp_col_checked

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('supp_col/', supp_col, name='supp_col'),
    path('supp_col_checked/', supp_col_checked, name='supp_col_checked'),
    path('delete-columns/', delete_selected_columns, name='delete_selected_columns'),
]
