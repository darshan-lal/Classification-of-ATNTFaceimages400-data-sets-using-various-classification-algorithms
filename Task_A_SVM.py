import numpy as np
from sklearn.svm import SVC


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
	traindata=train
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

dataset=pickDataClass('HandWrittenLetters.txt',letter_2_digit_convert('ABCDE'))
dataset=dataset.transpose()
number_per_class=39
testdata,traindata=splitData2TestTrain(dataset,number_per_class,'1:20')#first 20 training last 19 testing


trainX=traindata[1:,:].transpose()
trainY=traindata[0,:]
testX=testdata[1:,:].transpose()
testY=testdata[0,:]



classifier=SVC(kernel='linear')
classifier.fit(trainX,trainY)

prediction=classifier.predict(testX)

error = testY - prediction
TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100 
f=open('Classification output Accuracy1.txt','w')
f.write('%.4f Percent'%TestingAccuracy)
f.close()
np.savetxt('Classification output1.txt',prediction,fmt="%0.0f",delimiter=',')
