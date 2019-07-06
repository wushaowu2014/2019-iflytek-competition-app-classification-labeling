# 2019-iflytek-competition-app-classification-labeling
2019大数据应用分类标注挑战赛 baseline  
赛题地址：http://challenge.xfyun.cn/2019/gamedetail?type=detail/classifyApp  

## 赛事任务  
根据app的应用描述，预测它属于哪种app，比如便捷生活，游戏，通讯社交，阅读，工作求职， 影音娱乐，教育，出行旅游，工具，等等，属于nlp里的多分类问题。 感兴趣的可以继续研究  

这个baseline比较简单，直接利用词向量+RidgeClassifier预测，线上准确率45左右，和线下基本一致。

## Todo
对数据进行清洗，jieba分词，去停用词等等，建议用其他cnn模型试试
希望有所收获！   
注：其他赛题的 baseline 可以关注https://github.com/wushaowu2014?tab=repositories

# 更新......
版本baseline1.py在baseline.py的基础上加入了分词操作，线上分数直接76+，目前直接进前五，此版本仅供参考，后续应该可以更高......

