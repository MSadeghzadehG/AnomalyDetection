#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import csv
from operator import itemgetter
import numpy as np
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# from datetime import datetime

file5 = open('201811291521.txt', 'r')
dataset = file5.read()
print('data read')
dataset = dataset.split('\n')
data = []
for i in range(0, len(dataset)):
    data.append(dataset[i].split('|'))
data = sorted(data, key=itemgetter(0))
data.pop(0)
print('data sorted')

headings = ['stime', 'etime', 'sip', 'sport', 'sipint', 'mac', 'osname', 'osversion', 'fingerprint', 'dip', 'dport', 'dipint', 'dstmac', 'rosname', 'rosversion', 'rfingerprint', 'protocol', 'pkts', 'bytes', 'rpkts', 'rbytes', 'dur', 'iflags', 'riflags', 'uflags', 'ruflags', 'entropy', 'rentropy', 'tos', 'rtos', 'application', 'vlanint', 'domain', 'endreason', 'hash']
print('data : '+str(len(data)))
# for item in data:
#     item[0] = datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S.%f')
#     item[1] = datetime.strptime(item[1], '%Y-%m-%d %H:%M:%S.%f')
#     print(item[0].strftime('%m/%d/%Y'))

data = np.array(data)
df = pd.DataFrame(data)
df.columns = headings
print('dataset created')
print(df.head())
# print(df['protocol'].nunique())


# In[3]:


edited_df = df.drop(['stime','etime','sipint','mac','osname','osversion','fingerprint','dipint','dstmac','rosname','rosversion','rfingerprint','iflags','riflags','uflags','ruflags','entropy','rentropy','tos','rtos','application','vlanint','domain','hash','pkts','bytes','rpkts','rbytes','dur','endreason'],axis=1)
print(edited_df.head())
# Get one hot encoding of columns B
one_hot = pd.get_dummies(edited_df['protocol'])
# Drop column B as it is now encoded
edited_df = edited_df.drop('protocol',axis = 1)
headers = []
for i in one_hot.columns:
    headers.append('protocol_' + i)
# Join the encoded df

one_hot.columns = headers
# edited_df = edited_df.join(one_hot)

def correct_ip(s):
    o = ''
    if '.' in s:
        for part in s.split('.'):
            part = part.zfill(3)
            o += part 
    else:
        o = o.zfill(12)
    o = o[:3] + '.' + o[3:]
    o = o[:7] + '.' + o[7:]
    o = o[:11] + '.' + o[11:]
    return o

def correct_port(s):
    return(s.zfill(5))
        


sip_headers = []
dip_headers = []

for i in range(4):
    sip_headers.append('sip_'+str(i))
    dip_headers.append('dip_'+str(i))

sip = []
for ip in edited_df['sip']:
    sip.append(map(int,correct_ip(ip).split('.')))

dip = []
for ip in edited_df['dip']:
    dip.append(map(int,correct_ip(ip).split('.')))
        
sport = []
for port in edited_df['sport']:
    sport.append(int(port))
    
dport = []
for port in edited_df['dport']:
    dport.append(int(port))
# print(len(sip[0]))
# print(len(dip[0]))
# print(len(dport[0]))
# print(len(sport[0]))
8
sip_df = pd.DataFrame(sip,columns=sip_headers)
dip_df = pd.DataFrame(dip,columns=dip_headers)
sport_df = pd.DataFrame(sport,columns=['sport'])
dport_df = pd.DataFrame(sport,columns=['dport'])


result = pd.concat([sip_df, dip_df, sport_df, dport_df, one_hot], axis=1, sort=False)
result.head()
result.describe()
print(len(result.values))
# sport_df
# edited_df


# In[9]:


# from keras.layers import Input, Dense, Lambda
# from keras.models import Model
# from keras.objectives import binary_crossentropy
# from keras.callbacks import LearningRateScheduler

# import numpy as np
# import matplotlib.pyplot as plt
# import keras.backend as K
# import tensorflow as tf


# m = 6
# n_z = 2
# n_epoch = 10

# input_size = 17
# hidden_size = 8


# # Q(z|X) -- encoder
# inputs = Input(shape=(input_size,))
# h_q = Dense(hidden_size, activation='relu')(inputs)
# mu = Dense(n_z, activation='linear')(h_q)
# log_sigma = Dense(n_z, activation='linear')(h_q)

# def sample_z(args):
#     mu, log_sigma = args
#     eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
#     return mu + K.exp(log_sigma / 2) * eps


# # Sample z ~ Q(z|X)
# z = Lambda(sample_z)([mu, log_sigma])
# # P(X|z) -- decoder
# decoder_hidden = Dense(hidden_size, activation='relu')
# decoder_out = Dense(input_size, activation='sigmoid')

# h_p = decoder_hidden(z)
# outputs = decoder_out(h_p)

# # Overall VAE model, for reconstruction and training
# vae = Model(inputs, outputs)

# # Encoder model, to encode input into latent variable
# # We use the mean as the output as it is the center point, the representative of the gaussian
# encoder = Model(inputs, mu)

# # Generator model, generate new data given latent variable z
# d_in = Input(shape=(n_z,))
# d_h = decoder_hidden(d_in)
# d_out = decoder_out(d_h)
# decoder = Model(d_in, d_out)

# def vae_loss(y_true, y_pred):
#     """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
#     # E[log P(X|z)]
#     recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
#     # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
#     kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

#     return recon + kl

# vae.compile(optimizer='adam', loss=vae_loss)
# vae.fit(result, result, batch_size=m, nb_epoch=n_epoch)


# In[ ]:


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


x_train,x_test = train_test_split(result, test_size=0.2)
print(x_train.shape, x_test.shape)

# kf = KFold(n_splits=2) # Define the split - into 2 folds 
# kf.get_n_splits(x_train) # returns the number of splitting iterations in the cross-validator
# print(kf) 
# KFold(n_splits=2, random_state=None, shuffle=False)

# loo = LeaveOneOut()
# loo.get_n_splits(X)



input_d = Input(shape=(17,))
encoded = Dense(14, activation='relu')(input_d)
encoded = Dense(10, activation='relu')(encoded)
encoded = Dense(6, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

decoded = Dense(6, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(decoded)
decoded = Dense(14, activation='relu')(decoded)
decoded = Dense(17, activation='sigmoid')(decoded)

autoencoder = Model(input_d, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])




n_folds = 10
fold_len = int(len(x_train)/n_folds)
cross_validation_scores = []
for i in range(n_folds):
    print("Running Fold", i+1, "/", n_folds)
    fold_train = [k for k in range(0,i*fold_len)] + [k for k in range((i+1)*fold_len,len(x_train))]
    fold_test = [item for item in range(i*fold_len,(i+1)*fold_len)]
    test = x_train.iloc[fold_test]
    train = x_train.iloc[fold_train]
    autoencoder.fit(train, train,
            epochs=10,
            batch_size=256,
            shuffle=True,
            validation_data = (test,test))
    fold_validation_score = autoencoder.evaluate(test,test,verbose=0)
    print("fold validation score = ", fold_validation_score)
    cross_validation_scores.append(fold_validation_score[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cross_validation_scores), np.std(cross_validation_scores)))
autoencoder.evaluate(x=x_test,y=x_test)    
    
# autoencoder.fit(x_train, x_train,
#             epochs=100,
#             batch_size=256,
#             shuffle=True,
#             validation_split = 0.2)



# In[57]:


list_of_loss = []#autoencoder.predict()
print(result.iloc[0])
for index, row in result.iterrows():
    print(result.iloc[index].values)
    list_of_loss.append(autoencoder.evaluate(row,row,verbose=0))


# In[ ]:




