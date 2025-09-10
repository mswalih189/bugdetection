from django.shortcuts import render
from django.contrib.auth.models import User,auth 
from django.http import HttpResponseRedirect
from django.contrib import messages
from .models import BugPrediction
# Create your views here.
def index(request):
    return render(request,"index.html")



def register(request):
    if request.method=="POST":
        first=request.POST['fname']
        last=request.POST['lname']
        uname=request.POST['uname']
        em=request.POST['email']
        ps=request.POST['psw']
        ps1=request.POST['psw1']
        if ps==ps1:
            if User.objects.filter(username=uname).exists():
                messages.info(request,"Username Exists")
                return render(request,"register.html")
            elif User.objects.filter(email=em).exists():
                messages.info(request,"Email exists")
                return render(request,"register.html")
            else:
                user=User.objects.create_user(first_name=first,
            last_name=last,username=uname,email=em,password=ps)
                user.save()
                return HttpResponseRedirect("login")
        else:
            messages.info(request,"Password not Matching")
            return render(request,"register.html")

    return render(request,"register.html")

def login(request):
    if request.method=="POST":
        uname=request.POST['uname']
        ps=request.POST['psw']
        user=auth.authenticate(username=uname,password=ps)
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('data')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"login.html")
    return render(request,"login.html")

def adminlogin(request):
    if request.method=="POST":
        un=request.POST['uname']
        ps=request.POST['psw']
        user=auth.authenticate(username=un,password=ps)
        if user.is_superuser is not None:
            auth.login(request,user)
            return HttpResponseRedirect('adminhome')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"adminlogin.html")
    return render(request,"adminlogin.html")

def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')


def data(request):
    if request.method=="POST":
        bugid=request.POST['bugid']
        comp=request.POST['component']
        product=request.POST['product']
        short=request.POST['short']
        status=request.POST['status']
        code=float(request.POST['code'])
        bug=float(request.POST['bug'])
        sev=float(request.POST['severity'])
        votes=float(request.POST['votes'])
        comments=float(request.POST['comments'])
        import pandas as pd
        df=pd.read_csv(r"static/Dataset/BugDetect.csv")
        print(df.head())
        print(df.isnull().sum())
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        bugid1=l.fit_transform([bugid])
        comp1=l.fit_transform([comp])
        product1=l.fit_transform([product])
        short1=l.fit_transform([short])
        status1=l.fit_transform([status])
        print(df.dropna(inplace=True))
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        bugid_1=l.fit_transform(df["bug_id"])
        component_1=l.fit_transform(df["component_name"])
        product_1=l.fit_transform(df["product_name"])
        desc_1=l.fit_transform(df["short_description"])
        status_1=l.fit_transform(df["status_category"])
        df["Component_Name"]=component_1
        df["Product_Name"]=product_1
        df["Description"]=desc_1
        df["Status"]=status_1
        df["BugID"]=bugid_1
        df=df.drop(["bug_id","long_description","assignee_name","reporter_name","resolution_category","resolution_code"],axis=1)
        df=df.drop(["component_name","product_name","short_description","status_category"],axis=1)
        X=df.drop("severity_category",axis=1)
        y=df["severity_category"]
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.45)

        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression()
        log.fit(X_train,y_train)
        pred_log=log.predict(X_test)
        from sklearn.metrics import accuracy_score
        log_acc=accuracy_score(pred_log,y_test)
        print("Logistic Regression Accuracy Score: ",log_acc)
        from sklearn.naive_bayes import GaussianNB
        gn=GaussianNB()
        gn.fit(X_train,y_train)
        pred_gn=gn.predict(X_test)
        log_gn=accuracy_score(pred_gn,y_test)
        print("Naive Bayes Accuracy Score: ",log_gn)
        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier()
        rf.fit(X_train,y_train)
        pred_rf=rf.predict(X_test)
        log_rf=accuracy_score(pred_rf,y_test)
        print("Random Forest Accuracy Score: ",log_rf)
        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier()
        knn.fit(X_train,y_train)
        pred_knn=gn.predict(X_test)
        log_knn=accuracy_score(pred_knn,y_test)
        print("KNN Accuracy Score: ",log_knn)
        from sklearn.svm import SVC
        svc=SVC()
        svc.fit(X_train,y_train)
        pred_svc=svc.predict(X_test)
        log_svc=accuracy_score(pred_svc,y_test)
        print("SVM Accuracy Score: ",log_svc)
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        def plot_confusion_matrix(cm, title='CONFUSION MATRIX', cmap=plt.cm.Reds):
            target_names=['Blocker','Normal',"Major","Minor","Critical","Trivial"]
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        confusionMatrix = confusion_matrix(y_test, pred_log)
        print('Confusion matrix, Logisitc Regression')
        print(confusionMatrix)
        plot_confusion_matrix(confusionMatrix)
        plt.show()
        
        
        from sklearn.naive_bayes import GaussianNB
        rf=GaussianNB()
        rf.fit(X,y)
        import numpy as np
        data_1=np.array([[comp1,product1,short1,status1,code,bug,sev,bugid1,votes,comments]],dtype=object)
        prediction_data=rf.predict(data_1)
        print(prediction_data)
        bg=BugPrediction.objects.create(component_name=comp,product_name=product,short_desc=short,
        status_category=status,status_code=code,bug_fix=bug,severity_code=sev,bug_severity=prediction_data)
        bg.save()
        return render(request,"predict.html",{"comp":comp,"product":product,"short":short,"status":status,
        "code":code,"bug":bug,"sev":sev,"prediction_data":prediction_data})

    return render(request,"data.html")


def predict(request):
    return render(request,"predict.html")


def adminhome(request):
    bg=BugPrediction.objects.all()
    return render(request,"adminhome.html",{"bg":bg})
