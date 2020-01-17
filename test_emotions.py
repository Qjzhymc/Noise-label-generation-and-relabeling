from __future__ import print_function,absolute_import,division,with_statement
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from classification import LearningWithNoisyLabels
import latent_estimation
import warnings
warnings.simplefilter("ignore")
np.random.seed(477)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from scipy.io import arff
#这个是划分了数据集的，后来说不需要划分测试集，所以使用了test_emotions_2.py
if __name__=='__main__':
	#1.获取数据集，添加噪声标签
	dataset=arff.loadarff('emotions.arff')#数据593行，78列，
	data = pd.DataFrame(dataset[0],dtype=np.float) #获取emotions.arff文件中的data属性的数据，前面都是说明属性的文字不需要
	data=data.values
	X=data[:,:-6]#提取前72列作为特征
	y=data[:,-1:]#提取最后1列作为添加噪声的标签
	z=data[:,-6:-1]#其他不需要动的特征加入z

	y=[int(label) for label in y]#将数组中的每个str类型转化为int
	print(type(y[0]))

	#0.0先划分训练集测试集
	train_x=X[:500,:]
	train_y=y[:500]
	train_z=z[:500,:]
	test_x=X[501:,:]
	test_y=y[501:]
	test_z=z[501:,:]
	#先将测试数据转化为dataframe
	test_x_data=pd.DataFrame(test_x)
	test_y_data=pd.DataFrame(test_y)
	test_z_data=pd.DataFrame(test_z)
	#再将x,z,y合并为test_data
	test_data=pd.concat([test_x_data,test_z_data],axis=1)
	test_data=pd.concat([test_data,test_y_data],axis=1)

	test_data.to_csv('emotions_test_data.csv',index=None)#先将test_data写入测试数据文件中
	
	#再将训练数据转化为dataframe
	train_x_data=pd.DataFrame(train_x)
	train_y_data=pd.DataFrame(train_y)
	train_z_data=pd.DataFrame(train_z)
	#再将x,z,y合并为train_data
	train_data=pd.concat([train_x_data,train_z_data],axis=1)
	train_data=pd.concat([train_data,train_y_data],axis=1)
	#train_data还要加入s再写入训练文件



	#噪声率=0.1 噪声样本=50  0.836 0.826
	#0.2 100  0.793 0.760
	#0.3 150  0.728 0.663
	#0.4 200  0.586 0.532
	#0.5 250  0.510 0.445
	#0.6 300  0.347 0.402
	#0.7 350  0.228 0.293
	#0.8 400  0.206 0.250
	#0.9 450  0.184 0.217
	NUM_ERRORS=450
	s=np.array(train_y)
	error_indices=np.random.choice(len(s),NUM_ERRORS,replace=False)
	for i in error_indices:
		wrong_label=np.random.choice(np.delete(range(2),s[i]))
		s[i]=wrong_label
	actual_label_errors=np.arange(len(train_y))[s!=train_y]
	#train_data加入s，并写入文件
	train_data['s']=s
	train_data.to_csv('emotions_train_data.csv',index=None)
	print('\n Indices of actural label errors:\n',actual_label_errors)
	print('hello')
	#2.获取预测概率矩阵
	psx=latent_estimation.estimate_cv_predicted_probabilities(train_x,s,clf=LogisticRegression(max_iter=1000,multi_class='auto',solver='lbfgs'))

	#K=len(np.unique(s))
	#3.使用噪声训练的模型测试
	clf=LogisticRegression(solver='lbfgs',multi_class='auto')
	baseline_score=accuracy_score(test_y,clf.fit(train_x,s).predict(test_x))
	print("使用X_train和s训练的逻辑回归：",baseline_score) 
	#4.使用重标注测试
	rp=LearningWithNoisyLabels()
	rp_score=accuracy_score(test_y,rp.fit(train_x,s,psx=psx).predict(test_x))
	print("使用X_train和s和psx的逻辑回归(+rankpruning)：",rp_score)