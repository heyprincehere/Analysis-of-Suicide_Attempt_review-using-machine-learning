import os
from django.shortcuts import render
from Users.models import UserRegistrationModel # type: ignore
from assets import *
from django.contrib import messages

# Create your views here.
def registerUser(request):
    if request.method=='POST':
        name=request.POST['name']
        loginid=request.POST['loginid']
        pswd=request.POST['pswd']
        email=request.POST['email']
        state=request.POST['state']
        location=request.POST['location']
        mobile=request.POST['mobile']
        

        ur=UserRegistrationModel(name=name,loginid=loginid,password=pswd,email=email,state=state,location=location,mobile=mobile)
        #form=UserRegistrationForm(request.POST)
        if ur:
            print('Data is Valid')
            ur.save()
            messages.success(request, 'You have been successfully registered')
            
            return render(request, 'UserLogin.html')
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            return render(request, 'UserRegistrations.html')
            print("Invalid form")
    else:
      
        return render(request, 'UserRegistrations.html')

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "Activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})
        
import pandas as pd
from django.conf import settings 
from .data_preprocessing import main, prediction_value
   
def dataset(request):
   
    path = os.path.join(settings.MEDIA_ROOT, 'media', 'train.csv')
    df = pd.read_csv(path, nrows=100)
    df = df.to_html()
    return render(request, 'users/datasetreview.html', {'data': df})

def prediction(request):
    return render(request,'users/dataprediction.html')

def Classification_result(request):
    accurecy,precission,recall=main()
    return render(request,'users/view_classification_result.html',context={'accuracy':accurecy})
    
def suicide_predict(request):
    if request.method == 'POST':
        country = request.POST['country']
        year=int(request.POST['year'])
        age=request.POST['age']
        #suicide=int(request.POST['suicide_no'])
        
        mb= {
                'country': country,
                'year':year,
                'age': age,
                #'suicides_no':suicide
            }

        result = prediction_value(mb)
        return render(request, 'users/dataprediction.html', {'country': country,'year':year,'age':age, 'result': result})
    else:
        return render(request, 'users/test_results.html', {})
