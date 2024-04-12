from django.urls import path
from .views import process_csv
from .views import delete_missing_row
from .views import delete_selected_columns
from .views import imputate_selected_column
from .views import linear_regression

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('delete_missing_row/', delete_missing_row, name='delete_missing_row'),
    path('delete_selected_columns/', delete_selected_columns, name='delete_selected_columns'),
    path('imputate_selected_column/', imputate_selected_column, name='imputate_selected_column'),
    path('linear_regression/', linear_regression, name='linear_regression'),
]
