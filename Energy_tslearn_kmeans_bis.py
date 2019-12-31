# -*- coding: utf-8 -*-
"""
Created on Wed May 22 05:30:43 2019

@author: Youssef AAMER
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:46:40 2019

@author: Youssef AAMER
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt


from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
seed = 0
numpy.random.seed(seed)

path    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Exemple_energy_hourly_All_Net.xlsx'
Dataset = pd.read_excel(path,headers=None)
path2    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Exemple_Trafic_hourly_All_Net.xlsx'
Dataset2 = pd.read_excel(path2,headers=None)
Number_Clusters=4
a=Dataset.shape[0]
b=Dataset.shape[1]-1

path1    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Liste_sites_bis.xlsx'
site_list = pd.read_excel(path1)

Dataset_N=Dataset.loc[:, Dataset.columns.difference(['Site'])].fillna(Dataset.median())
Dataset2_N=Dataset2.loc[:, Dataset2.columns.difference(['Site'])].fillna(Dataset2.median())

X_train = Dataset_N.as_matrix()

X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[0:a])  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=b).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]

print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=Number_Clusters, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(Number_Clusters):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means Energy Clustring")

        
Result=pd.DataFrame()
Result['site']=Dataset['Site']
Result['Cluster_Energy']=y_pred
Result['Total_Energy_by_Traff']=Dataset.sum(axis = 1, skipna = True)
df=Result.groupby('Cluster_Energy').sum()
df1=Result.groupby('Cluster_Energy').count()
df1['Sum_Energy_by_Traff']=df['Total_Energy_by_Traff']
df1['Energy_by_Traff_mean_by_cluster'] = df1['Sum_Energy_by_Traff']/df1['site']
print(df1)



X_train2 = Dataset2_N.as_matrix()

X_train2 = TimeSeriesScalerMeanVariance().fit_transform(X_train2[0:a])  # Keep only 50 time series
X_train2 = TimeSeriesResampler(sz=b).fit_transform(X_train2)  # Make time series shorter
sz2 = X_train2.shape[1]

print("Euclidean k-means")
km2 = TimeSeriesKMeans(n_clusters=Number_Clusters, verbose=True, random_state=seed)
y_pred2 = km2.fit_predict(X_train2)

plt.figure()
for yi in range(Number_Clusters):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train2[y_pred2 == yi]:
        
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km2.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz2)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means Traffic Clustring")
        

Result['Cluster_Trafic']=y_pred2
Result['Total_Traffic']=Dataset2_N.sum(axis = 1, skipna = True)

Result2=pd.DataFrame()
Result2['Total_Traffic']=Dataset2.sum(axis = 1, skipna = True)
Result2['site']=site_list
Result2['Cluster_Trafic']=y_pred2
df2=Result2.groupby('Cluster_Trafic').sum()
df3=Result2.groupby('Cluster_Trafic').count()
df3['Sum_Trafic']=df2['Total_Traffic']
df3['Trafic_mean_by_cluster'] = df3['Sum_Trafic']/df3['site']
print(df3)
Result.to_excel('Clustering__casa_all.xlsx')
df1.to_excel('Clustering__casa_synthese_energy.xlsx')
df3.to_excel('Clustering__casa_synthese_trafic.xlsx')
df1.sort_values(by='Energy_by_Traff_mean_by_cluster', axis=0, ascending=True, inplace=True)
df1.reset_index(inplace=True)
a=df1.loc[df1['Sum_Energy_by_Traff'] == df1['Sum_Energy_by_Traff'].max(), 'Cluster_Energy'].iloc[0]
Cluster_to_analyze = Result.loc[Result['Cluster_Energy'] == a]
Cluster_to_analyze["Good_site"] = ""

#----------------------LOOP to get nearest site---------------
A=Cluster_to_analyze.shape[0]

for i in range(A):
   Site_1=Cluster_to_analyze.iat[i,0]
   Trafic_site1=Cluster_to_analyze.iat[i,4]
   b=Result.loc[Result['site'] == Site_1, 'Cluster_Trafic'].iloc[0]
   temp_df=Result.loc[Result['Cluster_Trafic'] == b]
   temp_df1=temp_df.loc[Result['Cluster_Energy'] == df1.iat[0,0]]
   temp_df1['delta_trafix_abs']=abs(temp_df1['Total_Traffic']-Trafic_site1)
   temp_df1.sort_values(by='delta_trafix_abs', axis=0, ascending=True, inplace=True)
   Good_site1=temp_df1.iat[0,0]
   Cluster_to_analyze.iat[i,5]=Good_site1

Cluster_to_analyze.to_excel('Cluster_to_analyze.xlsx')

path3    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Site_Parameters.xlsx'
Site_Parameters1 = pd.read_excel(path3,headers=None)
Site_Parameters=Site_Parameters1.fillna(-1000)
#path4    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Cluster_to_analyze_clean.xlsx'
#Cluster_to_analyze_clean = pd.read_excel(path4,headers=None)

from sklearn import preprocessing
from scipy.spatial import distance
min_max_scaler = preprocessing.MinMaxScaler()
Site_Parameters_clean=Site_Parameters.loc[:, Site_Parameters.columns != 'Site']
Site_Parameters_clean = Site_Parameters_clean.values.astype(float)
Site_Parameters_clean = min_max_scaler.fit_transform(Site_Parameters_clean)
Site_Parameters_clean = pd.DataFrame(Site_Parameters_clean)
Site_col=Site_Parameters['Site']
Site_Parameters_clean.insert(loc=0, column='Site', value=Site_col)

Final_Result = pd.DataFrame(numpy.zeros((Site_Parameters_clean.shape[1]-1,Cluster_to_analyze.shape[0] )))

B = Cluster_to_analyze.shape[0]

for j in range(B):
   Le_site=Cluster_to_analyze.iat[j,0]
   V1 = Site_Parameters_clean.loc[Site_Parameters_clean['Site'] == Cluster_to_analyze.iat[j,0]]
   X=V1.T
   new_header = X.iloc[0]
   X = X[1:]
   X.columns = new_header
   V2 = Site_Parameters_clean.loc[Site_Parameters_clean['Site'] == Cluster_to_analyze.iat[j,5]]
   Y=V2.T
   new_header2 = Y.iloc[0]
   Y = Y[1:]
   Y.columns = new_header2
   Z=pd.concat([X, Y], axis=1)
   Z[Le_site]=abs(Z.iloc[ : , 1 ]-Z.iloc[ : , 0 ])
   Final_Result.iloc[:,j]=Z[Le_site]
   Final_Result.rename(columns={Final_Result.columns[j]:Le_site}, inplace=True)

Final_Result.to_excel('Final_Result.xlsx')

#-------------Définition de la liste des paraùetres recomandé-----------
#---------------------------------------------------------------------------

Good_Cluster=Result.loc[Result['Cluster_Energy'] == df1.iat[0,0]]

C = Good_Cluster.shape[0]

mask = Site_Parameters['Site'].isin(Good_Cluster['site'])
Site_Parameters_Good_Cluster=Site_Parameters[mask]
C = Site_Parameters_Good_Cluster.shape[1]-1
Parameters_Template = pd.DataFrame(numpy.zeros((Site_Parameters_Good_Cluster.shape[1]-1 ,2)))
List_Parameters=Site_Parameters_Good_Cluster.columns.values
#Site_Parameters['Cell radius(m)--1'].value_counts().head(1)
Parameters_Template.rename(columns={Parameters_Template.columns[0]:'Parameter'}, inplace=True)
Parameters_Template.rename(columns={Parameters_Template.columns[1]:'Value'}, inplace=True)


for k in range(C):
    
   Parameter_name=List_Parameters[k+1]
   Value=Site_Parameters[Parameter_name].value_counts().head(1)
   Value2 = Value.to_frame(name=None)
   Recommanded_value=Value2.index[0]
   Parameters_Template.iloc[k,0]=Parameter_name
   Parameters_Template.iloc[k,1]=Recommanded_value

#-------------Application des parametres recommandés aux site bad energy Cluster-----------
#---------------------------------------------------------------------------  
Parameters_Template.index += 1 
Bad_cluster_delat_parameter=Final_Result.T
Bad_cluster_delat_parameter.columns=List_Parameters[1:]
Bad_cluster_delat_parameter = Bad_cluster_delat_parameter.reset_index()
Bad_cluster_delat_parameter.rename(columns={'index': 'Site'}, inplace=True)

mask1 = Site_Parameters['Site'].isin(Bad_cluster_delat_parameter['Site'])
Site_Parameters_Bad_Cluster=Site_Parameters[mask1]
New_Site_Parameters_Bad_Cluster=Site_Parameters_Bad_Cluster

New_Parameters_Bad_Cluster = pd.DataFrame(numpy.zeros((1 ,Site_Parameters_Bad_Cluster.shape[1] )))
New_Parameters_Bad_Cluster.columns=List_Parameters[:]
New_Parameters_Bad_Cluster.rename(columns={'index': 'Site'}, inplace=True)
#
#Nouvelle_site_parametrage=pd.DataFrame()
#Nouvelle_site_parametrage['Parameter']=Parameters_Template['Parameter']
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
#Nouvelle_site_parametrage.drop('index', inplace=True, axis=1)
#
#for l in range(B):
#    
#   Le_site2=Bad_cluster_delat_parameter.iat[l,0]
#   S1 = New_Site_Parameters_Bad_Cluster.loc[New_Site_Parameters_Bad_Cluster['Site'] == Le_site2]
#   S2=S1.append(Bad_cluster_delat_parameter.loc[l,:])
#   S2=S2.reset_index()
#   S2.drop('index', inplace=True, axis=1)
#   S2.loc[1,'Site']='Delta'
#   S3=S2.T
#   headers = S3.iloc[0]
#   S3  = pd.DataFrame(S3.values[1:], columns=headers)
#   Para= Parameters_Template.reset_index()
#   S3['Parameter']=Para['Value']
#   S3['Name_Parameter']=Para['Parameter']
#
#   for m in range(0,C-1):
#      if S3.iat[m,1]!= 0 and S3.iat[m,0]!=-1000 and S3.iat[m,2]!=-1000:
#          S3.iat[m,0]=S3.iat[m,2]
#   Nouvelle_site_parametrage[Le_site2] =S3[Le_site2]
#
#
#Nouvelle_site_parametrage.set_index('Parameter',inplace=True)
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.T
#Nouvelle_site_parametrage.to_excel('Nouvelle_site_parametrage.xlsx')
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
#Nouvelle_site_parametrage.rename(columns={'index': 'Site'}, inplace=True)


#----------------------------Deep Learning for energy saving prediction------------------
#--------------------------------------------------------------------------------
pathx    = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Site_Parameters_Casa.xlsx'
Site_Parameters_Casa = pd.read_excel(pathx,headers=None)

path4_features = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Data_set_for_DL_energy.xlsx'
Data_set_for_DL_energy = pd.read_excel(path4_features,headers=None)

path4_Output = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Output_DL_energy.xlsx'
Output_DL_energy = pd.read_excel(path4_Output,headers=None)

Data_set_for_DL_energy_F=pd.merge(Data_set_for_DL_energy,Site_Parameters_Casa,on='Site')
Data_set_for_DL_energy_F.to_csv('Data_set_for_DL_energy_Final.csv')

import keras as keras
import tensorflow as tf
print(tf.__version__)
from keras.initializers import normal
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
initializer = normal(mean=0, stddev=0.02, seed=30)

x_train, x_test, y_train, y_test = train_test_split(Data_set_for_DL_energy_F.iloc[: , 1:1367], Output_DL_energy.iloc[:,1], test_size=0.2, random_state=42)
input_dim1 = x_train.shape[1]
#
print('comment: Dataset splited')
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#
#model_Energy = keras.models.Sequential()
#model_Energy.add(keras.layers.Dense(400,input_dim=input_dim1, activation='relu'))
#model_Energy.add(keras.layers.Dense(400, activation='relu'))
#model_Energy.add(keras.layers.Dense(400, activation='relu'))
#model_Energy.add(keras.layers.Dense(1,activation='linear'))
#
#model_Energy.compile(optimizer='adam',
#              loss='mae',
#              metrics=['mse','mae'])
#
#model_Energy.fit(x_train, y_train,validation_data=(x_test, y_test),epochs=1500)
#val_loss_Energy , val_acc_Energy,val_acc_Energy = model_Energy.evaluate(x_test, y_test)
#
#model_Energy.save('Energy_Usage.model')
#new_model = keras.models.load_model('Energy_Usage.model')


#----------------------------------------------------------
Cluster_to_analyze_B=Cluster_to_analyze.rename(columns={'site': 'Site'})


Cluster_to_analyze_B.drop(['Cluster_Energy','Total_Energy_by_Traff','Cluster_Trafic','Total_Traffic'], inplace=True, axis=1)
Bad_cluster_delat_parameter_B=pd.merge(Bad_cluster_delat_parameter,Cluster_to_analyze_B,on='Site')

Nouvelle_site_parametrage=pd.DataFrame()
Nouvelle_site_parametrage['Parameter']=Parameters_Template['Parameter']
Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
Nouvelle_site_parametrage.drop('index', inplace=True, axis=1)

for l in range(B):
    
   Le_site3=Bad_cluster_delat_parameter_B.iat[l,0]
   Le_site4=Bad_cluster_delat_parameter_B.iat[l,1365]
   S5 = New_Site_Parameters_Bad_Cluster.loc[New_Site_Parameters_Bad_Cluster['Site'] == Le_site3]
   S6=S5.append(Bad_cluster_delat_parameter.loc[l,:])
   S6=S6.reset_index()
   S6.drop('index', inplace=True, axis=1)
   S6.loc[1,'Site']='Delta'
   S7=S6.T
   headers2 = S7.iloc[0]
   S7  = pd.DataFrame(S7.values[1:], columns=headers2)
   S4 = Site_Parameters.loc[Site_Parameters['Site'] == Le_site4].T
   header1 = S4.iloc[0]
   S4 = S4[1:]
   S4=S4.rename(columns = header1)
   S4=S4.reset_index()
   S7['Parameter_Good_site']=S4[Le_site4]
   S7['Name_Parameter']=S4['index']

   for m in range(0,C-1):
      if S7.iat[m,1]!= 0 and S7.iat[m,0]!=-1000:
          S7.iat[m,0]=S7.iat[m,2]
   Nouvelle_site_parametrage[Le_site3] =S7[Le_site3]


Nouvelle_site_parametrage.set_index('Parameter',inplace=True)
Nouvelle_site_parametrage=Nouvelle_site_parametrage.T
Nouvelle_site_parametrage.to_excel('Nouvelle_site_parametrage.xlsx')
Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
Nouvelle_site_parametrage.rename(columns={'index': 'Site'}, inplace=True)

#----------------------------------------------------------------------------------

#mask2 = Data_set_for_DL_energy['Site'].isin(Nouvelle_site_parametrage['Site'])
#Data_set_for_DL_energy_Bad_Cluster=Data_set_for_DL_energy[mask2]
#Data_set_for_DL_energy_Bad_Cluster_F=pd.merge(Data_set_for_DL_energy_Bad_Cluster,Nouvelle_site_parametrage,on='Site')
#Data_set_for_DL_energy_Bad_Cluster.to_excel('Data_set_for_DL_energy_Bad_Cluster.xlsx')
#predictions=pd.DataFrame()
#Data_need_energy=Data_set_for_DL_energy_Bad_Cluster_F.iloc[: , 1:1367]
#Data_need_energy = scaler.transform(Data_need_energy)
#predictions = new_model.predict(Data_need_energy)
#numpy.savetxt("predictions_bad-cluster.csv", predictions, delimiter=",")

pathy_features = 'D:\\Maching Learning\\Energy Efficency\\KPIs Inwi\\Data_set_for_DL_energy_Net.xlsx'
Data_set_for_DL_energy_Net = pd.read_excel(pathy_features,headers=None)

new_model = keras.models.load_model('D:\Maching Learning\Energy Efficency\Python\Energy_Usage.model')

mask2 = Data_set_for_DL_energy_Net['Site'].isin(Nouvelle_site_parametrage['Site'])
Data_set_for_DL_energy_Bad_Cluster=Data_set_for_DL_energy_Net[mask2]
Data_set_for_DL_energy_Bad_Cluster_F=pd.merge(Data_set_for_DL_energy_Bad_Cluster,Nouvelle_site_parametrage,on='Site')
Data_set_for_DL_energy_Bad_Cluster.to_excel('Data_set_for_DL_energy_Bad_Cluster.xlsx')
predictions=pd.DataFrame()
Data_need_energy=Data_set_for_DL_energy_Bad_Cluster_F.iloc[: , 1:1367]
Data_need_energy = scaler.transform(Data_need_energy)
predictions = new_model.predict(Data_need_energy)
numpy.savetxt("predictions_bad-cluster.csv", predictions, delimiter=",")

#----------------------------------------------------------
#Cluster_to_analyze_B=Cluster_to_analyze.rename(columns={'site': 'Site'})
#
#
#Cluster_to_analyze_B.drop(['Cluster_Energy','Total_Energy_by_Traff','Cluster_Trafic','Total_Traffic'], inplace=True, axis=1)
#Bad_cluster_delat_parameter_B=pd.merge(Bad_cluster_delat_parameter,Cluster_to_analyze_B,on='Site')
#
#Nouvelle_site_parametrage=pd.DataFrame()
#Nouvelle_site_parametrage['Parameter']=Parameters_Template['Parameter']
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
#Nouvelle_site_parametrage.drop('index', inplace=True, axis=1)
#
#for l in range(B):
#    
#   Le_site3=Bad_cluster_delat_parameter_B.iat[l,0]
#   Le_site4=Bad_cluster_delat_parameter_B.iat[l,1365]
#   S5 = New_Site_Parameters_Bad_Cluster.loc[New_Site_Parameters_Bad_Cluster['Site'] == Le_site3]
#   S6=S5.append(Bad_cluster_delat_parameter.loc[l,:])
#   S6=S6.reset_index()
#   S6.drop('index', inplace=True, axis=1)
#   S6.loc[1,'Site']='Delta'
#   S7=S6.T
#   headers2 = S7.iloc[0]
#   S7  = pd.DataFrame(S7.values[1:], columns=headers2)
#   S4 = Site_Parameters.loc[Site_Parameters['Site'] == Le_site4].T
#   header1 = S4.iloc[0]
#   S4 = S4[1:]
#   S4=S4.rename(columns = header1)
#   S4=S4.reset_index()
#   S7['Parameter_Good_site']=S4[Le_site4]
#   S7['Name_Parameter']=S4['index']
#
#   for m in range(0,C-1):
#      if S7.iat[m,1]!= 0 and S7.iat[m,0]!=-1000:
#          S7.iat[m,0]=S7.iat[m,2]
#   Nouvelle_site_parametrage[Le_site3] =S7[Le_site3]
#
#
#Nouvelle_site_parametrage.set_index('Parameter',inplace=True)
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.T
#Nouvelle_site_parametrage.to_excel('Nouvelle_site_parametrage.xlsx')
#Nouvelle_site_parametrage=Nouvelle_site_parametrage.reset_index()
#Nouvelle_site_parametrage.rename(columns={'index': 'Site'}, inplace=True)
