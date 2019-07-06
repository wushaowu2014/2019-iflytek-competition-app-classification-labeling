# -*- coding: utf-8 -*-
"""
@author: shaowu

任务：给定一个app，根据它的应用描述，去预测它的主要功能，比如是属于体育,或游戏，或旅游，等等
Todo:
    进一步清洗数据，比如去掉停用词，模型可以尝试cnn应该比较好,可以去参考 2017知乎看山杯
"""

import pandas as pd
import numpy as np
import time
import datetime
import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
import jieba
# 分词处理
def split_discuss(data):
    data['length'] = data['Discuss'].apply(lambda x:len(x))
    data['Discuss'] = data['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))
    
    return data
##读入app类型标签对应表，第一列为编码，第二列是具体的含义：
apptype_id_name= pd.read_csv("apptype_id_name.txt",sep='\t',header=None)
apptype_id_name.columns=['label_code','label']
print(apptype_id_name.nunique())

#============================读入训练集：=======================================
train= pd.read_csv("apptype_train.dat",header=None,encoding='utf8',delimiter=' ')
#以tab键分割，不知道为啥delimiter='\t'会报错，所以先读入再分割。
train=pd.DataFrame(train[0].apply(lambda x:x.split('\t')).tolist(),columns=['id','label','conment'])

#=============================读入测试集：======================================
test= pd.read_csv("app_desc.dat",header=None,encoding='utf8',delimiter=' ')
test=pd.DataFrame(test[0].apply(lambda x:x.split('\t')).tolist(),columns=['id','conment'])
print('数据读入完成！')
print('训练集标签分布：',train.label.value_counts())

#========================以|为分隔符，把标签分割：===============================
train['label1']=train['label'].apply(lambda x:x.split('|')[0])
train['label2']=train['label'].apply(lambda x:x.split('|')[1] if '|' in x else 0) ##第二个标签有些没有，此处补0
print('训练集第一个标签分布：',train.label1.value_counts())
'''
可以发现第一个标签有125个，相当于125类多分类问题，而且不平衡问题挺严重的！下面三类少于5个样本，这里不考虑，后续可以考虑
140110       4
140805       3
140105       1
'''
##去掉样本少于5个的类别,（主要考虑到后续的5折交叉验证）：
train=train[~train.label1.isin(['140110','140805','140105'])].reset_index(drop=True)

#===========================下面以第一个标签训练模型=============================
##分词：
train['conment'] = train['conment'].apply(lambda x:' '.join(jieba.cut(x)))
test['conment'] = test['conment'].apply(lambda x:' '.join(jieba.cut(x)))
#tf-idf特征：
column='conment'
vec = TfidfVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8,use_idf=1,smooth_idf=1, sublinear_tf=1) #这里参数可以改
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
print(trn_term_doc.shape)
##下面对标签进行编码：
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
lbl.fit(train['label1'].values)
train['label1'] = lbl.transform(train['label1'].values)
label=train['label1']
num_class=train['label1'].max()+1


#=======================模型训练：5折交叉验证=========================================
n_folds=5
stack_train = np.zeros((train.shape[0],num_class))
stack_test = np.zeros((test.shape[0],num_class))
for i, (tr, va) in enumerate(StratifiedKFold(label, n_folds=n_folds, random_state=42)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    
    ridge = RidgeClassifier(random_state=42)
    ridge.fit(trn_term_doc[tr], label[tr])
    score_va = ridge._predict_proba_lr(trn_term_doc[va])
    score_te = ridge._predict_proba_lr(test_term_doc)
    
    stack_train[va] += score_va
    stack_test += score_te
    
    
print("model acc_score:",metrics.accuracy_score(label,np.argmax(stack_train,axis=1), normalize=True, sample_weight=None))

##获取第一第二个标签：取概率最大的前两个即可：
m=pd.DataFrame(stack_train)
first=[]
second=[]
for j,row in m.iterrows():
    zz=list(np.argsort(row))
    first.append(row.index[zz[-1]]) ##第一个标签
    second.append(row.index[zz[-2]]) ##第二个标签
m['label1']=first
m['label2']=second

#计算准确率，只要命中一个就算正确：
k=0
for i in range(len(label)):
    if label[i] in [m.loc[i,'label1'],m.loc[i,'label2']]:
        k+=1
    else:
        pass
print('线下准确率：%f'%(k/len(label)))

##准备测试集结果：
results=pd.DataFrame(stack_test)
first=[]
second=[]
for j,row in results.iterrows():
    zz=list(np.argsort(row))
    first.append(row.index[zz[-1]]) ##第一个标签
    second.append(row.index[zz[-2]]) ##第二个标签
results['label1']=first
results['label2']=second
##之前编码，最后逆编码回来：
results['label1']=results['label1'].apply(lambda x:lbl.inverse_transform(int(x)))
results['label2']=results['label2'].apply(lambda x:lbl.inverse_transform(int(x)))

##结合id列，保存：
pd.concat([test[['id']],results[['label1','label2']]],axis=1).to_csv('submit.csv',index=None,encoding='utf8')
