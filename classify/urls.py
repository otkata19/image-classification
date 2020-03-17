from django.urls import path

from . import views

app_name = 'classify'

urlpatterns = [
    path('test/', views.ClassifyView.as_view(), name='index'),
]
