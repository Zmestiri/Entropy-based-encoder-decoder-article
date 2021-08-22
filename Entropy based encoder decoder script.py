# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:47:37 2021

@author: zmestiri
"""
##################### Import libraries #######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
sns.set_style("whitegrid")
import os
np.random.seed(697)

##################Import data ################################################

rootdir2=r"C:\Users\zmestiri\OneDrive - Dover Corporation\Documents\MIBUSINESS\Work\Anomaly detection\Try\A2.3.csv"
nfile=pd.read_csv(rootdir2)
df=nfile[['X1','X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']]


##################################Pre-processing##############################
for col in df.columns : 
            df.loc[:,col] = pd.to_numeric(df.loc[:,col])
df.reset_index(drop=True, inplace=True)
df.isnull().sum() #No missing values thus no imputations needed
      #### standardize data ####
x=df[['X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']].values
import numpy as np
from sklearn.preprocessing import StandardScaler
datastandard = StandardScaler().fit_transform(x)
coll=['X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']
norm_data=pd.DataFrame(datastandard, columns=coll)
norm_data.reset_index(drop=True, inplace=True)
norm_data['X1']=df['X1'].values
      ####Split in 75% train and 25% test set####
train, test_df = train_test_split(norm_data, test_size = 0.15, random_state= 1984)
train_df, dev_df = train_test_split(train, test_size = 0.15, random_state= 1984)
train_x =np.array(train_df[['X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']])
dev_x =np.array(dev_df[['X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']])
test_x = np.array(test_df[['X2', 'X3','X4', 'X5', 'X6','X7', 'X8', 'X9','X10', 'X11','X12', 'X13']])

######################Building the AutoEncoder###################################

# Choose size of our encoded representations (we will reduce the initial features to this number)
encoding_dim = 3
# Define input layer
input_data = Input(shape=(train_x.shape[1],))
# Define encoding layer
encoded = Dense(encoding_dim, activation='elu')(input_data)
# Define decoding layer
decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)
# Create the autoencoder model
autoencoder = Model(input_data, decoded)
#Compile the autoencoder model
autoencoder.compile(optimizer='adam',
                    loss='mse')
#Fit to train set, validate with dev set and save to hist_auto for plotting purposes
hist_auto = autoencoder.fit(train_x, train_x,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(dev_x, dev_x))

# Summarize history for loss
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
encoder = Model(input_data, encoded)

# Create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Encode and decode our test set (compare them vizually just to get a first insight of the autoencoder's performance)
encoded_x = encoder.predict(test_x)
decoded_output = decoder.predict(encoded_x)

##################Build new model using encoded data##########################
#Encode data set from above using the encoder
encoded_train_x = encoder.predict(train_x)
encoded_test_x = encoder.predict(test_x)

encoderoutput= pd.DataFrame(encoded_test_x)
encoderoutput1= pd.DataFrame(encoded_train_x)
encoderoutput1['index']=train_df['Time Stamp Control'].values
encoderoutput['index']=test_df['Time Stamp Control'].values
encoderoutput= pd.concat([encoderoutput,encoderoutput1])
encoderoutput.reset_index(drop=True, inplace=True)
#reccuperation of the original input 
inputorig= pd.DataFrame(test_df)
input1orig= pd.DataFrame(train_df)
inputorig= pd.concat([inputorig,input1orig])
inputorig.reset_index(drop=True, inplace=True)

#calculation of the signature matrix and entrepy based on time window
liste=[]
liste2=[]
for i in range(1,len(hola),200):
    j=i+200
    liste2=liste2+[j]
    dfff=hola.iloc[i:j]
    #signature matrix
    import seaborn as sns
    corr_mat=dfff.corr(method='spearman')
    Matcorr=sns.clustermap(corr_mat) 
    #Von Neumann entropy calculation 
    import numpy as np
    from numpy import linalg as LA
    e, v = LA.eig(corr_mat)
    t = e * np.log(e)
    liste=liste+[-np.sum(t)]
#save the output for drift analysis
EnTdata=pd.DataFrame({})
EnTdata['time window']=liste2
EnTdata['Entropy']=liste
ch=r'C:\Users\zmestiri\OneDrive - Dover Corporation\Documents\MIBUSINESS\Work\Anomaly detection\Try\Results\EntropystudyA2.3.xlsx'
EnTdata.to_excel(ch, index=False)
#threshold for anomaly extruction
k=np.mean(EnTdata['Entropy'])
k1=np.std(EnTdata['Entropy'])
st=k-3*k1
wh=EnTdata[EnTdata['Entropy']> st ]
wh.reset_index(drop=True, inplace=True)
#decode data
result=pd.DataFrame()
h=encoderoutput[['F0','F1','F2']]
for i in range(len(wh)):
    numb=wh['time window'][i]
    decoded_output = decoder.predict(h.iloc[numb-200:numb])
    sequence = pd.DataFrame(decoded_output)
    sequence.reset_index(drop=True, inplace=True)
    please=encoderoutput['index'].iloc[numb-200:numb]
    sequence['index']=please.values
    result= pd.concat([result,sequence])
result.reset_index(drop=True, inplace=True)
final=pd.DataFrame({})
# saving the result                                   
for i in range(len(result)):
    j=result['index'][i]
    ligne=inputorig[inputorig['Time Stamp Control'] == j].index.tolist()
    ligne1=inputorig.iloc[ligne]
    final= pd.concat([final,ligne1])
ch=r'C:\Users\zmestiri\OneDrive - Dover Corporation\Documents\MIBUSINESS\Work\Anomaly detection\Try\Results\output of decoderalpha2.3.csv'
final.to_csv(ch, index= 'F')
##### END #####
