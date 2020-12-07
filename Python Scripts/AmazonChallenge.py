import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from collections import Counter
import csv
from xgboost import XGBClassifier

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",20)

df = pd.read_csv("F:/ML_Projects/AmazonRecommendation/train.csv")



#-------------some pre data processing(quite obvious ones)---------------
df.drop('customer_id',axis=1,inplace=True)
n_columns_numerical = df.shape[1] - 3   #since last 3 are categorical
columns = df.columns.values

# ---------Exploratory Data Analysis--------------

# For numerical predictors
# for i in range(n_columns_numerical):
#     col_name = columns[i]
#     plt.hist(df.iloc[:,i])
#     plt.title(col_name)
#     plt.show()

#Last 2 categorical predictors
# for j in range(2):
#     sizes = df.iloc[:,n_columns_numerical+j].value_counts()
#     plt.pie(sizes.tolist(),autopct='%1.2f%%',labels=sizes.index.values)
#     plt.title(columns[n_columns_numerical+j])
#     plt.show()

# sbn.countplot(hue='customer_category',x='customer_active_segment',data=df)
# plt.show()

# fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout= True)
# i = 0
# for ax in axs.flat:
#     ax.boxplot(x=columns[i],data=df)
#     ax.set_xlabel(columns[i])
#     i += 1
# sbn.boxplot(x='customer_product_search_score',data=df)
# plt.show()

# #-------------------correlation between numeric variables and target--------------
# numeric_data = df.select_dtypes(include=[np.number])
# corr_numeric = numeric_data.corr()
# sbn.heatmap(corr_numeric,cmap="YlGnBu",annot=True)
# plt.xticks(rotation=45)
# plt.show()    #--this gives us info that cust_stay and cust_ctr score are highly correlated

#let's remove one of those
# df.drop('customer_stay_score',inplace=True,axis=1)    #this dropped my acuracy

#least correlated with target - customer_affinity_score, customer_product_search_score

# df.drop(['customer_affinity_score','customer_product_search_score'],inplace=True,axis=1)

#------Treat Missing Values-------   #it's imroving accuracy
df['customer_product_search_score'].fillna(df['customer_product_search_score'].mean(),inplace=True)
df['customer_stay_score'].fillna(df['customer_stay_score'].median(),inplace=True)
df['customer_product_variation_score'].fillna(df['customer_product_variation_score'].median(),inplace=True)
df['customer_order_score'].fillna(df['customer_order_score'].mean(),inplace=True)

#-------Hot encoding categorical variables------
df = pd.get_dummies(df,prefix=['ActiveSegment','X1'],dummy_na=True)


#-----Data split---------
# df = df.dropna()
x = df.drop('customer_category',axis=1)
y = df['customer_category']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# #------Before Upsampling------
# classifier = RandomForestClassifier(n_estimators = 50, random_state = 0)
# classifier.fit(x_train, y_train)
# predict_val_rf = classifier.predict(x_test)
#
# print('------Random Forest Before upsampling---------')
# print("Accuracy : ", accuracy_score(y_test, predict_val_rf) *  100)
# print("Recall : ", recall_score(y_test, predict_val_rf) *  100)
# print("Precision : ", precision_score(y_test, predict_val_rf) *  100)
# print(confusion_matrix(y_test, predict_val_rf))
# print(classification_report(y_test, predict_val_rf))
# print("\n\n\n\n")
#
#
# #-------Upsampling-------

#
# print("Before Upsampling:-")
# print(Counter(y_train))
# X_train_upsampled, y_train_upsampled = resample(x_train[y_train == 1],
#                                                 y_train[y_train == 1],
#                                                 replace=True,
#                                                 n_samples=x_train[y_train == 0].shape[0],
#                                                 random_state=123)
#
# X_train_upsampled = np.vstack((x_train[y_train==0], X_train_upsampled))
# y_train_upsampled = np.hstack((y_train[y_train==0], y_train_upsampled))
# print("After Upsampling:-")
# print(Counter(y_train_upsampled))
#
#
# classifier = RandomForestClassifier(n_estimators = 50, random_state = 0)
# classifier.fit(X_train_upsampled, y_train_upsampled)
# predict_val_rf = classifier.predict(x_test)
#
# print('------Random Forest After upsampling---------')
# print("Accuracy : ", accuracy_score(y_test, predict_val_rf) *  100)
# print("Recall : ", recall_score(y_test, predict_val_rf) *  100)
# print("Precision : ", precision_score(y_test, predict_val_rf) *  100)
# print(confusion_matrix(y_test, predict_val_rf))
# print(classification_report(y_test, predict_val_rf))
# print("\n\n\n\n")


#---------------Processing OF TEST DATA-------------------------
test_df = pd.read_csv("F:/ML_Projects/AmazonRecommendation/test.csv")
user_id = test_df['customer_id'].tolist()
test_df.drop('customer_id',axis=1,inplace=True)
# n_columns_numerical = test_df.shape[1] - 2   #since last 3 are categorical
# columns = test_df.columns.values

# print(test_df.info())
# missing_cols = [1,3,5,6]
# for i in missing_cols:
#     plt.hist(test_df.iloc[:,i])
#     plt.title(test_df.columns.values[i])
#     plt.show()

#------Treat Missing Values-------
test_df['customer_product_search_score'].fillna(test_df['customer_product_search_score'].mean(),inplace=True)
test_df['customer_stay_score'].fillna(test_df['customer_stay_score'].median(),inplace=True)
test_df['customer_product_variation_score'].fillna(test_df['customer_product_variation_score'].median(),inplace=True)
test_df['customer_order_score'].fillna(test_df['customer_order_score'].mean(),inplace=True)

#-------Hot encoding categorical variables------
test_df = pd.get_dummies(test_df,prefix=['ActiveSegment','X1'],dummy_na=True)




#----Upsampling using SMOTE on training data------
from imblearn.over_sampling import SMOTE
oversample = SMOTE()



for i in range(1):
    # x_train, y_train = oversample.fit_resample(x,y)     #don't do sampling, it reduces accuracy

    # print("Before Upsampling:-")
    # print(Counter(y_train))
    # print("After Upsampling:-")
    # print(Counter(y_train_upsampled))

    # classifier = RandomForestClassifier(n_estimators = 50, random_state = 0)
    # classifier.fit(X_train_upsampled, y_train_upsampled)
    # predict_val_rf = classifier.predict(x_test)

    # --XGBoost---
    model = XGBClassifier(learning_rate=0.3,max_depth=3,subsample=0.9,gamma=0.2,min_child_weight=2.27)
    #keep max_depth = 3, learning_rate=0.3, subsamlpe=0.9, gamma = 0.2, eta = default, colsample_bytree = default, min_child_weight=2.27
    model.fit(x,y)
    # predict_val_rf2 = model.predict(x_test)
    # print(confusion_matrix(y_test, predict_val_rf2))
    # print(classification_report(y_test, predict_val_rf2))    #giving accuracy of 97% and recall of 86%


    #----Logistic Regression----

    # #standardization
    # numeric_data_col = df.select_dtypes(include=[np.number]).columns.values[:8]
    # print(numeric_data_col)
    # for i in numeric_data_col:
    #     scale = StandardScaler().fit(X_train_upsampled[[i]])
    #     X_train_upsampled[i] = scale.transform(X_train_upsampled[[i]])
    #     x_test[i] = scale.transform(x_test[[i]])
    # model2 = LogisticRegression()
    # model2.fit(X_train_upsampled,y_train_upsampled)
    # predict_val_rf3 = model2.predict(x_test)
    # print(confusion_matrix(y_test,predict_val_rf3))
    # print(classification_report(y_test,predict_val_rf3))

    # print('------Random Forest After upsampling using SMOTE---------')
    # print("Accuracy : ", accuracy_score(y_test, predict_val_rf) *  100)
    # print("Recall : ", recall_score(y_test, predict_val_rf2) *  100)
    # print("Precision : ", precision_score(y_test, predict_val_rf) *  100)
    # print(confusion_matrix(y_test, predict_val_rf))
    # print(classification_report(y_test, predict_val_rf))
    # print("\n\n\n\n")



    #
    predict_val_rf = model.predict(test_df)
    # # #
    pd.DataFrame({'customer_id': user_id, 'customer_category': predict_val_rf}).to_csv("F:/ML_Projects/AmazonRecommendation/result1Dec_2.csv", index=False)
    # # # with open("F:/ML_Projects/AmazonRecommendation/result.csv", "a") as fp:
    # #     wr = csv.writer(fp, dialect='excel')
    # #     for val in predict_val_rf:
    # #         wr.writerow([val])
