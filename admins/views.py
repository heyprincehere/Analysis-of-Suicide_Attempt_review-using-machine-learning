from django.shortcuts import render
from django.contrib import messages

from Users.models import UserRegistrationModel

# Create your views here.
def adminLogin(request):
    if request.method=="POST":
       loginid=request.POST['loginid']
       pswd=request.POST['pswd']
       if loginid=='admin' and pswd=='admin':
           
           return render(request,'admin/adminHome.html')
       else:
            messages.error(request, 'Please enter details carefully')

            return render(request,'AdminLogin.html')

def userDetails(request):
    ud=UserRegistrationModel.objects.all()
    print(ud)
    return render(request,'admin/userDetais.html',context={'ud':ud})

def updateUserStatus(request):
    loginid=request.GET['loginid']
    usu=UserRegistrationModel.objects.get(loginid=loginid)
    if usu.status=='waiting':
        usu.status='Activated'
        usu.save()
        ud=UserRegistrationModel.objects.all()
        print(ud)
        return render(request,'admin/userDetais.html',context={'ud':ud})


    

