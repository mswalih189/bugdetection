from django.urls import path 

from .import views 

urlpatterns=[
    path('',views.index,name='index'),
    path('register',views.register,name='register'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('adminlogin',views.adminlogin,name='adminlogin'),
    path('adminhome',views.adminhome,name='adminhome'),
    path('data',views.data,name='data'),
    path('predict',views.predict,name='predict'),
]
