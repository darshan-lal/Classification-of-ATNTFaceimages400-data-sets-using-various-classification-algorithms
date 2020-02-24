#darshan_lal
#1001667684
#harsha_boppana
#1001667684
#ref:https://github.com/koreaditya/Face-Image-and-Handwritten-Letter-Image-Recognition-


from scipy.spatial import distance
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#ref:https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def Euclidean(A_train,B_test):
    dst = distance.euclidean(A_train, B_test)
    dist=dst**0.5    
    dist=('%.2f' % dist)
    return float(dist)

#Centroid
centroidList={}
def Centroid(train_Xdata,train_Ydata):
    centroidListArray=[]
    for o in range (0,len(train_Xdata[0])):
        sum=0
        for p in range(len(train_Xdata)):
            sum+=(train_Xdata[p][o])
        centroidListArray.append(float('%.2f'%(sum/len(train_Xdata))))
    a=train_Ydata[0]
    centroidList[a]=centroidListArray 

def pickDataClass(filename, class_ids):
    data=[]
    load_data=np.loadtxt(filename,delimiter=',')
    for i in range (len(load_data[0])):
        if load_data[0][i] in class_ids:
            data.append(load_data[:,i])
        else:continue;
        stdata=np.stack(data)        
    return(stdata)

def splitData2TestTrain(filename, number_per_class,  test_instances):
    max=number_per_class
    min=0
    test=[]
    train=[]
    instance=test_instances.split(':')
    temp=int(instance[0])
    temp1=int(instance[1])


    while max<=len(filename[0]):
        test.append(filename[:,temp:temp1])
        train.append(filename[:,min:temp])
        train.append(filename[:,temp1:max])
        temp+=number_per_class
        temp1+=number_per_class
        max+=number_per_class
        min+=number_per_class
            
            
    test=np.hstack(test)
    train=np.hstack(train)
    return(test,train)

def save(test,train):
    test_data=[]
    train_data=[]
    test_data=test
    train_set=train
    fp=open('testfile.txt', 'w')
    np.savetxt(fp, test_data)
    fp1=open('trainfile.txt', 'w')
    np.savetxt(fp1, train_data)
    fp.close()
    fp1.close()
    print("test data and train data is saved successfully")
    return(test,train)

def letter_2_digit_convert(string):
        letters = []
        for x in string:
            letters.append(ord(x)-64)
        letters=np.stack(letters)
        return(letters)

def classid_length(string):
    string1=string
    classidlen=len(letter_2_digit_convert(string1))
    return(classidlen)


string1='ABCDE'
lenclassids=classid_length(string1)
data=pickDataClass('HandWrittenLetters.txt',letter_2_digit_convert(string1))
data=data.transpose()
number_per_class=39
test_set,train_set=splitData2TestTrain(data,number_per_class,'39:30')#first 30 training last 9 testing

train_X=train_set[1:,:].transpose()
train_Y=train_set[0,:,None]
test_X=test_set[1:,:].transpose()
test_Y=test_set[0,:]

temp_e=temp_c=0
for z in range(lenclassids-1):
    while train_Y[temp_e]==train_Y[temp_e+1]:
        temp_e+=1
    Centroid(train_X[temp_c:temp_e+1,0:],train_Y[temp_e]) #number each label
    temp_e+=1
    temp_c=temp_e
Centroid(train_X[temp_e:,0:],train_Y[-1])

centroi_dist=[]#distances of each testing instances from the centroids
for k in range(len(test_X)):
    centroi_dist.append({})
    for n in centroidList:
        centroi_dist[k][n]=Euclidean(centroidList[n],test_X[k])
        
prediction=[]
for n in centroi_dist:
    distl=min(n.values())
    for k,value in n.items():
        if value==distl:
            prediction.append(k)
            break
        
#Error test and accuracy
Error = test_Y - prediction
Accuracy_test = (1-np.nonzero(Error)[0].size/len(Error))*100 
f=open('Classification output Accuracy1.txt','w')
f.write('%.4f Percent'%Accuracy_test)
f.close()
np.savetxt('Classification output1.txt',prediction,fmt="%0.0f",delimiter=',')  
        
