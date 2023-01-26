from . import views
from django.urls import include, path


urlpatterns = [
    path('', views.index, name='index'),
    path('settings/', views.settings, name='settings'),
    path('search/', views.search, name='search'),
    # path('login/', views.signin, name='login'),
]