import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import numpy as np
import sklearn
from sklearn import datasets,metrics,tree
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation as cv
from sklearn import model_selection as cv # replaced the above line with this
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc 


#import the file as a pandas df
stalcDF = pd.read_csv("//kaggle-alcohol/student-mat.csv", encoding = "ISO-8859-1")





#first thing to do is pair up the data headers with the data in the first row
ob = stalcDF.iloc[0].reset_index().apply(tuple, axis=1)  #write array col head & row1 to a dict_items
colHeader_row1_list = [] 
for key, value in ob:
    list_item = (key,value)
    colHeader_row1_list.append(list_item)

categorical_cols = [] #this will be the list of categorical attributes
numeric_cols = []     #this will be the list of numeric attributes

for i, x in colHeader_row1_list:
#     print((type(x)),i,x) # just checking what kind of format the data is in (e.g., int or numpy.int64)
    if type(x) != str: 
        numeric_cols.append(i)
    else:
        categorical_cols.append(i)
        
print("There are:", (len(categorical_cols)), "text columns")
                            # 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                            # 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                            # 'nursery', 'higher', 'internet', 'romantic']
print("There are:", (len(numeric_cols)), "numeric columns")
                            # ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                            #  'failures', 'famrel', 'freetime', 
                            #  'goout', 'Dalc', 'Walc', 'health', 
                            #  'absences', 'G1', 'G2', 'G3']
        
        
#convert text in the columns with text data to numeric form via label encoding..
#which adds 17 new features to my set.
for i in categorical_cols:
#     print(i)
    stalcDF[i] = stalcDF[i].astype("category")
    stalcDF[(i+"_CAT")]  = stalcDF[i].cat.codes #duplicate each column encoded, and add "_CAT" after it
#len(stalcDF.columns) # prints = 50
        
  
 # #duplicate the DataFrame
stalcDF_N= stalcDF.copy() 

# print("Length of stalcDF_N before dropping textual columns:",len(stalcDF_N.columns)) #should print 50 on the first run
# for i in categorical_cols:
#     if i in stalcDF_N:
#         #print("dropping", i, "\n....")
#         stalcDF_N.drop([i], axis=1, inplace=True) #adapted the drop comment for this 
#         #print("dropped", i, "!")
#     else:
#         pass
# print("\nLength of stalcDF_N after dropping textual columns:",len(stalcDF_N.columns)) #should print 31

# #Len of new DF should be 33: 50 -17 (the len of textual columns identifed earlier)

#some feature work
stalcDF_N['Avg_Grade'] = stalcDF_N[['G1', 'G2','G3']].mean(axis=1) #https://stackoverflow.com/questions/48366506/calculate-new-column-as-the-mean-of-other-columns-pandas/48366525
stalcDF_N['Avg_Alc_Cnsmptn'] = stalcDF_N[['Dalc', 'Walc']].mean(axis=1) #https://stackoverflow.com/questions/48366506/calculate-new-column-as-the-mean-of-other-columns-pandas/48366525
#may come back and drop these columns or their base columns, during learning.
           
#Create my variables for the Algorithm: target_names, target, feature names and dataset.data



                                # Create the Target variables (names and numeric version)
#create "dataset.target_names"
stalcDF_target_names = str(list(set(stalcDF_N["sex"]))) #<class 'str'>                               
#create "dataset.target"
stalcDF_target = stalcDF_N["sex_CAT"].values # <class 'numpy.ndarray'>



                              #Features i.e., the dataset class label values as an array

#I want this to only have numeric data so create a numeric-only dataset. I can do this by dropping all columns that 
#match the column name of my "categorical_cols" variable i created earlier


#drop highly correlated columns 
stalcDF_N.drop(['G1', 'G2','G3'], axis=1, inplace=True) #drop the 3 colums
stalcDF_N.drop(['Dalc', 'Walc'], axis=1, inplace=True) #drop the 3 colums

#drop any text/duplciated columns 

stalcDF_Numeric= stalcDF_N.copy() 
print("Length of stalcDF_N before dropping textual columns:",len(stalcDF_Numeric.columns)) #should print 47 on the first run
for i in categorical_cols:
    if i in stalcDF_Numeric:
        #print("dropping", i, "\n....")
        stalcDF_Numeric.drop([i], axis=1, inplace=True) #adapted the drop comment for this 
        #print("dropped", i, "!")
    else:
        pass
print("\nLength of stalcDF_N after dropping textual columns:",len(stalcDF_Numeric.columns)) #should print 30


#CREATE THE FEATURE SET FIRST. = dataset.data

stalcDF_dataset_data_df = stalcDF_Numeric.copy() #i dont want to edit the original 
stalcDF_dataset_data_df.drop(["sex_CAT"], axis=1, inplace=True)#could also drop the highlighy correlated attributes too but will hold off for now
stalcDF_dataset_data = stalcDF_dataset_data_df.to_numpy()


                                            #Build the SVM classifier.

#create "dataset.feature_names" 
stalcDF_feature_names = str(set(list(stalcDF_dataset_data_df.columns))) #<class 'str'>

df = stalcDF_N.copy() #duplicate the df
#not really needed but i created two new columns (target and target nameds, for the gender of each 
#row just to help wrap my head around the placement of variables)
df["target"]=df["sex_CAT"]
df["target_names"]=df["sex"]
df.columns


                                          #some simple visuals via the df

df_m = df[df.target_names =="M"] # create a df with only males
df_f = df[df.target_names =="F"] # create a df with only females
plt.xlabel("absences")
plt.ylabel("Avg_Grade")
plt.scatter(df_m["Avg_Grade"],df_m["absences"],color="green",marker=".",label="Males")
plt.scatter(df_f["Avg_Grade"],df_f["absences"],color="blue",marker="*",label="Females")


                                  #need numerics for svm so swap to numeric df
df = stalcDF_Numeric.copy()
X = df.drop(["sex_CAT"], axis = "columns") #features i.e., iris.data from my above guide
y = df.sex_CAT                             #label    i.e., iris.data from my above guide



                                    #train and test the model and then visualse with an AUC curve
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
model = SVC(C=2.5,kernel="linear")
model.fit(X_train,y_train)
model.score(X_test,y_test)
y_pred_svm = model.decision_function(X_test)
svm_fpr,svmn_tpr,threshold=roc_curve(y_test,y_pred_svm)
auc_svm = auc(svm_fpr,svmn_tpr)
plt.figure(figsize=(5,5),dpi=100)
plt.plot(svm_fpr,svmn_tpr, linestyle="-", marker = "*", label=("SVM = %0.3f" %auc_svm))
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()


