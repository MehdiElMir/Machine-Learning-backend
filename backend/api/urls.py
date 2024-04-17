from django.urls import path
from . import views

urlpatterns = [
    path('process-csv/', views.process_csv, name='process_csv'),
    path('delete_missing_row/', views.delete_missing_row, name='delete_missing_row'),
    path('delete_selected_columns/', views.delete_selected_columns, name='delete_selected_columns'),
    path('imputate_selected_column/', views.imputate_selected_column, name='imputate_selected_column'),
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('save_state/', views.save_state, name='save_state'),
    path('load_state/', views.load_state, name='load_state'),
    path('generate_plot_data/', views.generate_plot_data, name='generate_plot_data'),
    path('regression_linear_sckitlearn/', views.regression_linear_sckitlearn, name='regression_linear_sckitlearn'),
    path('regression_linear_3D/', views.regression_linear_3D, name='regression_linear_3D'),
    path('cross_validation/', views.cross_validation, name='cross_validation'),

]
