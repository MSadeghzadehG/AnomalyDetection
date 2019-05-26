from operator import itemgetter
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read_file():
    file = open('201811291521.txt', 'r')
    dataset = file.read()
#     print(dataset)
    print('data read')
    return dataset

    
def convert_to_df(dataset):
    dataset = dataset.split('\n')
    data = []
    for i in range(0, len(dataset)):
        data.append(dataset[i].split('|'))
    data = sorted(data, key=itemgetter(0))
    data.pop(0)
    print('data sorted')
    headings = ['stime', 'etime', 'sip', 'sport', 'sipint', 'mac', 'osname', 'osversion', 'fingerprint', 'dip', 'dport', 'dipint', 'dstmac', 'rosname', 'rosversion', 'rfingerprint', 'protocol', 'pkts', 'bytes', 'rpkts', 'rbytes', 'dur', 'iflags', 'riflags', 'uflags', 'ruflags', 'entropy', 'rentropy', 'tos', 'rtos', 'application', 'vlanint', 'domain', 'endreason', 'hash']
    print('data rows number: '+ str(len(data)))
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = headings
    print('dataset created')
    return df


def embedding(df):
    
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
    
    # print(df.head())
    edited_df = df.drop(['stime','etime','sipint','mac','osname','osversion','fingerprint','dipint','dstmac','rosname','rosversion','rfingerprint','iflags','riflags','uflags','ruflags','entropy','rentropy','tos','rtos','application','vlanint','domain','hash','pkts','bytes','rpkts','rbytes','dur','endreason'],axis=1)
    # print(edited_df.head())
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

    sip_headers = []
    dip_headers = []

    for i in range(4):
        sip_headers.append('sip_'+str(i))
        dip_headers.append('dip_'+str(i))

    sip = []
    for ip in edited_df['sip']:
        sip.append(map(int,correct_ip(ip).split('.')))
    #     sip.append(correct_ip(ip).split('.'))

    dip = []
    for ip in edited_df['dip']:
        dip.append(map(int,correct_ip(ip).split('.')))
    #     dip.append(correct_ip(ip).split('.'))



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
    
    sip_df = pd.DataFrame(sip,columns=sip_headers)
    dip_df = pd.DataFrame(dip,columns=dip_headers)
    sport_df = pd.DataFrame(sport,columns=['sport'])
    dport_df = pd.DataFrame(sport,columns=['dport'])
    
    result = pd.concat([sip_df, dip_df, sport_df, dport_df, one_hot], axis=1, sort=False)
    print(result.head())
    return result


def get_dataframe():
    original_dataframe = convert_to_df(read_file())
    return embedding(original_dataframe)


# get_dataframe()
