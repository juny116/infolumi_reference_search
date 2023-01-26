from . import views
from django.urls import include, path

urlpatterns = [
    path('', views.index, name='index'),
    path('setting/', views.setting, name='setting'),
    path('search/', views.search, name='search'),
]