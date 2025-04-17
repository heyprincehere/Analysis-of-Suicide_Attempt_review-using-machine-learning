
from django.shortcuts import render
from assets import * # type: ignore

def index(request):
    return render(request,'index.html')
def UserLogin(request):
    return render(request,'UserLogin.html')
def UserRegisterForm(request):
    return render(request,'UserRegistrations.html')
def AdminLogin(request):
    return render(request,'AdminLogin.html')
