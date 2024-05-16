from django.urls import path
from . import views
urlpatterns = [
    path('process-csv/', views.process_csv, name='process_csv'),
    path('delete_missing_row/', views.delete_missing_row, name='delete_missing_row'),
    path('delete_selected_columns/', views.delete_selected_columns, name='delete_selected_columns'),
    path('imputate_selected_column/', views.imputate_selected_column, name='imputate_selected_column'),
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('regression_linear_sckitlearn/', views.regression_linear_sckitlearn, name='regression_linear_sckitlearn'),
    path('regression_linear_3D/', views.regression_linear_3D, name='regression_linear_3D'),
    path('cross_validation/', views.cross_validation, name='cross_validation'),
    path('knn_classification/', views.knn_classification, name='knn_classification'),
    path('knn_regression/', views.knn_regression, name='knn_regression'),
    path('smote/', views.smote, name='smote'),
    path('undersampling/', views.undersampling, name='smote_undersampling'),
    path('generate_value_counts/', views.generate_value_counts, name='generate_value_counts'),
    path('decision_tree/', views.decision_tree, name='decision_tree'),

     

]
