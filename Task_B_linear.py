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

def draw(a,b):
        #string1='ABCDEFGHIJ'
        s=''
        s=s+str(a)
        s+=':'
        s+=str(b)
        string1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
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
        labelencoder_Y = LabelEncoder()
        train_Y[:,0]=labelencoder_Y.fit_transform(train_Y[:,0])
        onehotencoder=OneHotEncoder(categorical_features=[0])
        train_Y=onehotencoder.fit_transform(train_Y).toarray()
        train_Y=train_Y.transpose()
        #Xtest=testdata[:,:]
        #Ytest=testdataY[0,:]
        
        
        A_train=np.ones((1,len(train_X[0])))#change
        A_test=np.ones((1,len(test_Y)))#change
        
        
        Xtrain_padding = np.row_stack((train_X,A_train))
        Xtest_padding = np.row_stack((test_X,A_test))
        
        B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), train_Y.T)
        Ytest_padding = np.dot(B_padding.T,Xtest_padding)
        Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
        err_test_padding = test_Y - Ytest_padding_argmax
        TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
        return(Ytest_padding_argmax,TestingAccuracy_padding)


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