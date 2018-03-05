#泰泰尼克号之灾，目的是通过训练集train.csv来预测测试集test.csv的Survived值，0表示死亡，1表示存活
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.ensemble import BaggingRegressor
# from sklearn.model_selection import ShuffleSplit
# from sklearn import cross_validation
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
# train_x = train_dataSet['Pclass'].reshape(-1,1)
# train_y = train_dataSet['Survived'].reshape(-1,1)

# neigh = KNeighborsClassifier()
# neigh.fit(train_x,train_y)
# test = pd.read_csv('test.csv')['Pclass']
# # su = pd.read_csv('gender_submission.csv')
# Survived_0 = data_train.Survived.value_counts()[0]
# Survived_1 = data_train.Survived.value_counts()[1]
# print(Survived_1)
# print(data_train)
# fig = plt.figure()
# # plt.tight_layout(pad=1.5)
# fig.set(alpha=0.5)
#数据分析
# plt.subplot2grid((2,3),(0,0)) #2*3的子图
# data_train.Survived.value_counts().plot(kind='bar')
# plt.tight_layout(h_pad=0.5,w_pad=0.5)
# plt.title("获救情况(1为获救)")
# plt.ylabel('人数')
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.tight_layout(h_pad=0.5,w_pad=0.5)
# plt.title('乘客等级分布')
# plt.ylabel('人数')
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived,data_train.Age)
# plt.title('按年龄看获救分布(1表示获救)')
# plt.ylabel('年龄')
# plt.grid(b=True,which='major',axis='y') #y轴网格显示
# plt.subplot2grid((2,3),(1,0),colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.tight_layout(h_pad=0.5,w_pad=0.5)
# plt.xlabel('年龄')
# plt.ylabel('密度')
# plt.title('各等级的乘客年龄分布')
# plt.legend(('头等舱','2等舱','3等舱'),loc='best')
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.tight_layout(h_pad=0.5,w_pad=0.5)
# plt.ylabel('人数')
# plt.title('各登岸口上船人数统计')
#各等级乘客获救情况
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({'获救':Survived_1,'死亡':Survived_0})
# df.plot(kind='bar')
# plt.title('各等级乘客获救情况')
# plt.xlabel('乘客等级')
# plt.ylabel('人数')
#等级乘客获救情况
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({'男性':Survived_m,'女性':Survived_f})
# df.plot(kind='bar',stacked=True) #stacked表示在同一个柱子上显示
# plt.xlabel('性别')
# plt.ylabel('人数')
#根据舱等级和性别的获救情况
# plt.title('根据舱等级和性别的获救情况')
# ax1 = fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass !=3].value_counts().plot(kind='bar',label='1',color='#FA2479')
# ax1.set_xticklabels(['未获救','获救'],rotation=0)
# plt.legend(['女性/高级舱'],loc='best')
# ax2 = fig.add_subplot(142,sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass ==3].value_counts().plot(kind='bar',label='female, low class',color='pink')
# ax2.set_xticklabels(['未获救','获救'],rotation=0)
# plt.legend(['女性/低级舱'],loc='best')
# ax3 = fig.add_subplot(143,sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass !=3].value_counts().plot(kind='bar',label='3',color='lightblue')
# ax3.set_xticklabels(['未获救','获救'],rotation=0)
# plt.legend(['男性/高级舱'],loc='best')
# ax4 = fig.add_subplot(144,sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass ==3].value_counts().plot(kind='bar',label='4',color='steelblue')
# ax4.set_xticklabels(['未获救','获救'],rotation=0)
# plt.legend(['男性/低级舱'],loc='best')
# plt.tight_layout(h_pad=0.5,w_pad=0.5)
#各登录港口的获救情况
# Survivid_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survivid_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({'获救':Survivid_1,'死亡':Survivid_0})
# df.plot(kind='bar',stacked=True)
# plt.title('各登录港口的获救情况')
# plt.xlabel('登录港口')
# plt.ylabel('人数')

# g1 = data_train.groupby(['SibSp','Survived'])
# df1 = pd.DataFrame(g1.count()['PassengerId'])
# g2 = data_train.groupby(['Parch','Survived'])
# df2 = pd.DataFrame(g2.count()['PassengerId'])
# print(df1,df2)

# print(data_train.Cabin.value_counts())

# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df = pd.DataFrame({'有客舱':Survived_cabin,'无客舱':Survived_nocabin})
# df.plot(kind='bar',stacked=True)
# plt.xlabel('Cabin有无')
# plt.ylabel('人数')

# Survived_HFare = data_train.Survived[data_train.Fare>=32].value_counts()
# Survived_LFare = data_train.Survived[data_train.Fare<32].value_counts()
# Survived_HFare_0 = data_train.Survived[data_train.Fare>=32].value_counts()[0]
# Survived_HFare_1 = data_train.Survived[data_train.Fare>=32].value_counts()[1]
# Survived_LFare_0 = data_train.Survived[data_train.Fare<32].value_counts()[0]
# Survived_LFare_1 = data_train.Survived[data_train.Fare<32].value_counts()[1]
# print('高船票的死亡率',Survived_HFare_0/(Survived_HFare_0+Survived_HFare_1))
# print('高船票的存活率',Survived_HFare_1/(Survived_HFare_0+Survived_HFare_1))
# print('低船票的死亡率',Survived_LFare_0/(Survived_LFare_0+Survived_LFare_1))
# print('低船票的存活率',Survived_LFare_1/(Survived_LFare_0+Survived_LFare_1))
# df = pd.DataFrame({'高票价':Survived_HFare,'低票价':Survived_LFare})
# df.plot(kind='bar',stacked=True)
# plt.xlabel('票价高低')
# plt.ylabel('人数')

# plt.show()
#处理缺失值
def set_missing_ages(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print(type(know_age))
    # print(know_age)
    # print(type(know_age))
    # print(unknown_age.shape)
    y = know_age[:,0]
    X = know_age[:,1:]
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    predictedAges = rfr.predict(unknown_age[:,1::])
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df,rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    train_sizes, train_scores, test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    print(type(train_sizes))
    print(train_sizes)
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('训练样本数')
        plt.ylabel('得分')
        # plt.gca().invert_yaxis() #y坐标轴反回来
        plt.grid()
        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='b')
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='r')
        plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='训练集上得分')
        plt.plot(train_sizes,test_scores_mean,'o-',color='r',label='交叉验证集上得分')
        plt.legend(loc='best')
        plt.draw()
        plt.show()
        # plt.gca().invert_yaxis()
        midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
        diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
        return midpoint,diff
if __name__ =='__main__':
    data_train,rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    #将类值型转化为数值型
    dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
    df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
    df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
    #将幅度大的值转化为[-1,1]
    scaler = StandardScaler()
    age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']],age_scale_param)
    fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']],fare_scale_param)
    #提取特征字段,用正则
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    #用逻辑回归训练
    clf = linear_model.LogisticRegression(C=0.5,penalty='l2',tol=1e-8,max_iter=200,n_jobs=-1)
    bagging_clf = BaggingRegressor(clf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
    bagging_clf.fit(X,y)
    # clf.fit(X,y)
    # clf = linear_model.LogisticRegression()
    # clf.fit(X,y)
    # plot_learning_curve(clf,'学习曲线',X,y)
    # 处理测试集
    data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0
    tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    X = null_age[:,1:]
    predicteAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull(),'Age')] = predicteAges
    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')
    df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
    df_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
    #将幅度大的值转化为[-1,1]
    scaler = StandardScaler()
    age_scale_param = scaler.fit(df_test[['Age']])
    df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']],age_scale_param)
    fare_scale_param = scaler.fit(df_test[['Fare']])
    df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']],fare_scale_param)
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = bagging_clf.predict(test)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
    print(result)
    result.to_csv('predictions1.csv',index=False)
# print(pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)}))
#     cv = cross_validation.ShuffleSplit()
#     print(np.mean(cross_val_score(clf,X,y,cv=5)))

# split_train, split_test = train_test_split(df, test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass_.*')
# clf = linear_model.LogisticRegression(C=0.5,penalty='l2',tol=1e-8,max_iter=200,n_jobs=-1)
# clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
# test_df = split_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(test_df.as_matrix()[:,1:])
# print(test_df.as_matrix)
# print(classification_report(test_df.as_matrix()[:,0],predictions))
# origin_data_train = pd.read_csv('train.csv')
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_test[predictions != test_df.as_matrix()[:,0]]['PassengerId'].values)]
# print(bad_cases.info())
# print(type(predictions))
# print(predictions.shape)
# print(predictions)

