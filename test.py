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
# 100时，0.538 0.527
# 0.05时 0.525 0.527
# 0.06时 0.525 0.529
# 0.07时 0.512 0.531
# 0.08时 0.510 0.514
# 0.09时 0.514 0.536
# 0.1时 0.523 0.529
# 0.2时0.468 0.505
# 0.3时0.456 0.483
# 0.4时 0.459 0.481
# 0.5时 0.443 0.468
# 0.6时 0.410  0.432
# 0.7时 0.251  0.270
#测试工作票all_tickets文件添加噪声的程序
if __name__=='__main__':
	#1.获取数据集，添加噪声标签
	data=pd.read_csv('all_tickets.csv',dtype=str)
	data=pd.DataFrame()
	X=data['body']
	y=data['urgency']
	z=data['ticket_type']

	y=y.values#将series转化为数组
	y=[int(label) for label in y]#将数组中的每个str类型转化为int
	print(type(y[0]))


	#0.0先划分训练集测试集
	train_x=X[:48000]
	train_y=y[:48000]
	train_z=z[:48000]

	test_x=X[48001:]
	test_y=y[48001:]
	test_z=z[48001:]
	test_x_data=pd.DataFrame(test_x)
	test_y_data=pd.DataFrame(test_y)
	test_z_data=pd.DataFrame(test_z)
	test_data=pd.concat([test_x_data,test_z_data],axis=1)
	test_data=pd.concat([test_data,test_y_data],axis=1)
	print(test_data.head())

	test_data.to_csv('test_data.csv',index=None)
	#0.1先用原来所有数据X训练tfidf 
	tfidf_vect=TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)
	tfidf_vect.fit(X)
	#0.2再将train中的body和ticket_type加入train_data
	train_data=pd.DataFrame({'body':train_x,'ticket_type':train_z})
	#0.3再将train_data转化为向量
	train_x=tfidf_vect.transform(train_x)
	test_x=tfidf_vect.transform(test_x)


# #4800 0.1
# #24000 0.5
# 	NUM_ERRORS=4800
# 	s=np.array(train_y)
# 	error_indices=np.random.choice(len(s),NUM_ERRORS,replace=False)
# 	for i in error_indices:
# 		wrong_label=np.random.choice(np.delete(range(4),s[i]))
# 		s[i]=wrong_label
# 	actual_label_errors=np.arange(len(train_y))[s!=train_y]
# 	#1.1再加入s，并写入文件
# 	train_data['s']=s
# 	train_data.to_csv('train_data.csv',index=None)
# 	print('\n Indices of actural label errors:\n',actual_label_errors)
# 	print('hello')
# 	#2.获取预测概率矩阵
# 	psx=latent_estimation.estimate_cv_predicted_probabilities(train_x,s,clf=LogisticRegression(max_iter=1000,multi_class='auto',solver='lbfgs'))

# 	#K=len(np.unique(s))
# 	#3.使用噪声训练的模型测试
# 	clf=LogisticRegression(solver='lbfgs',multi_class='auto')
# 	baseline_score=accuracy_score(test_y,clf.fit(train_x,s).predict(test_x))
# 	print("使用X_train和s训练的逻辑回归：",baseline_score) 
# 	#4.使用重标注测试
# 	rp=LearningWithNoisyLabels()
# 	rp_score=accuracy_score(test_y,rp.fit(train_x,s,psx=psx).predict(test_x))
# 	print("使用X_train和s和psx的逻辑回归(+rankpruning)：",rp_score)