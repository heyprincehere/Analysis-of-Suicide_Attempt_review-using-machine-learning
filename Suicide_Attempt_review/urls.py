"""
URL configuration for Suicide_Attempt_review project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Suicide_Attempt_review import views as mainview

from Users import views as ur
from admins import views as av


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mainview.index,name='index'),
    path('UserLogin',mainview.UserLogin,name='UserLogin'),
    path('UserRegisterForm',mainview.UserRegisterForm,name='UserRegisterForm'),
    path('AdminLogin',mainview.AdminLogin,name='AdminLogin'),


    #userurls
    path('UserRegister',ur.registerUser,name='registerUser'),
    path('UserLoginCheck',ur.UserLoginCheck,name="UserLoginCheck"),
    path('dataset',ur.dataset,name="dataset"),
    path('prediction',ur.prediction,name='prediction'),
    path('Classification_result',ur.Classification_result,name='Classification_result'),
    path('suicide_predict',ur.suicide_predict,name='suicide_predict'),

    

    #admin urls
    path('AdminLoginCheck',av.adminLogin,name='adminLogin'),
    path('userDetails',av.userDetails,name='userDetails'),
    path('updateUserStatus',av.updateUserStatus,name='updateUserStatus'),
   
    


    
    
]
