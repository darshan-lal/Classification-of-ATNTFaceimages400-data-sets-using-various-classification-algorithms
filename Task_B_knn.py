#darshan_lal
#1001667684
#harsha_boppana
#1001667684
#ref:https://github.com/koreaditya/Face-Image-and-Handwritten-Letter-Image-Recognition-

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

def election(classifiermatrixarrays):
    d={}
    for elements in classifiermatrixarrays:
        if elements in d:
            d[elements]+=1
        else:
            d[elements]=1
    for k,val in d.items():
        if val == max(d.values()):
            return(int(k))

def draw(a,b):
        #string1='ABCDEFGHIJ'
        s=''
        s=s+str(a)
        s+=':'
        s+=str(b)
        string1=[1,2,3,4,5]
        lenclassids=len(string1)
        data=pickDataClass('ATNTFaceImages400.txt',string1)
        data=data.transpose()
        number_per_class=10
        test_set,train_set=splitData2TestTrain(data,number_per_class,s)#first 30 training last 9 testing
                                                            
        
        train_X=train_set[1:,:].transpose()
        train_Y=train_set[0,:,None]
        test_X=test_set[1:,:].transpose()
        test_Y=test_set[0,:]
        
        #ref:https://github.com/koreaditya/Face-Image-and-Handwritten-Letter-Image-Recognition-
        #getting the distances
        distance1=[]
        distance_dict=[]
        classifiermatrix=[]
        for i in range(0,len(test_X)):
            classifiermatrix.append([])
            
            distance_dict.append({})
            distance1.append([])
            for j in range(0,len(train_X)):
                #key=distance
                #value=label
                distance_dict[i][Euclidean(train_X[j],test_X[i])]=train_Y[j]#here
                distance1[i].append(Euclidean(train_X[j],test_X[i]))
            distance1[i].sort()
        
        #election to select the maximum number of occurence
        
        
        #taking the k=3
        
        for k in range(len(distance1)):
            #classifiermatrix.append([])
            for l in range(5): #enter the value of k nearest neighbours
                a=distance1[k][l]
                classifiermatrix[k].append(distance_dict[k][a])
        finalclassification=[] #final classified elements
        for elements in classifiermatrix:
            finalclassification.append(election(elements))
            
        #error test and accuracy
        error = test_Y - finalclassification
        TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100
        return(finalclassification,TestingAccuracy)


def drew(a,b,sr):
    x,y=draw(a,b)
    str1="Classification output "+str(sr)
    np.savetxt(str1,x,fmt="%0.0f")
    str2="Output Accuracy "+str(sr) 
    f1=open(str2,'w')
    f1.write('%f'%y)
    f1.close()
    return(y)

accuracy1=drew(0,2,1)
accuracy2=drew(2,4,2)
accuracy3=drew(4,6,3)
accuracy4=drew(6,8,4)
accuracy5=drew(8,10,5)
AvgAccuracy=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5
f=open('AverageAccuracy','w')
f.write('%f'%AvgAccuracy)
f.close()