import tensorflow as tf 
import numpy as np
from sklearn import preprocessing
from tensorflow.contrib.learn.python.learn.datasets.base import *
import random
from indicator import *

batch_index = 0
lastprice_index = 0
volume_index = 1
totalprice_index = 2
buy_index = 4
sell_index = 3

para = 10
def load_csv_data_mse(filename,target_col,size,n_features):
    data_file = csv.reader(open(filename,'rb'))
    data = np.empty((size-1, n_features))
    target = np.empty((size-1,20))
    for i,ir in enumerate(data_file):
        del ir[0]
        del ir[0]
        del ir[0]
        if data_file.line_num ==1:
            data[i]=np.asarray(ir,dtype=np.float32)
        elif data_file.line_num == size:
            labellist=[]
            for il in target_col:
                labellist.append(ir[il])
            target[i-1]=np.asarray(labellist,dtype=np.float32)
        else:
            data[i]=np.asarray(ir,dtype=np.float32)
            labellist=[]
            for il in target_col:
                labellist.append(ir[il])
            target[i-1]=np.asarray(labellist,dtype=np.float32)
    data_scaled=preprocessing.scale(data)
    target_scaled = preprocessing.scale(target)
    print target_scaled
    return Dataset(data=data,target=target)

def get_class(sell,buy,last_sell,last_buy):
    if sell> last_sell:
        if buy > last_buy:
            return [1,0,0,0,0,0,0,0,0]
        elif abs(buy-last_buy)<0.5:
            return [0,1,0,0,0,0,0,0,0]
        elif buy < last_buy:
            return [0,0,1,0,0,0,0,0,0]
    elif abs(sell-last_sell)<0.5:
        if buy > last_buy:
            return [0,0,0,1,0,0,0,0,0]
        elif abs(buy-last_buy)<0.5:
            return [0,0,0,0,1,0,0,0,0]
        elif buy < last_buy:
            return [0,0,0,0,0,1,0,0,0]
    elif sell < last_sell:
        if buy > last_buy:
            return [0,0,0,0,0,0,1,0,0]
        elif abs(buy-last_buy)<0.5:
            return [0,0,0,0,0,0,0,1,0]
        elif buy < last_buy:
            return [0,0,0,0,0,0,0,0,1]

def get_class_commodity(data,i,sell_index,buy_index,size):
    index = i+1
    while index < size-1:
        if abs(data[i][sell_index]-0)<0.1 :
            data[i][sell_index] = data[i][buy_index]
            return [1,0]
        elif abs(data[i][buy_index]-0)<0.1:
            data[i][buy_index] = data[i][sell_index]
            return [0,1]
        if data[i][sell_index] > data[index][sell_index] or data[i][buy_index]> data[index][buy_index]:
            return [1,0]
        elif data[i][sell_index] < data[index][sell_index] or data[i][buy_index] < data[index][buy_index]:
            return [0,1]
        else:
            index += 1
    return [0,1]


def load_csv_data_softmax(filename,size,n_features):
    data_file = csv.reader(open(filename,'rb'))
    data = np.empty((size-1, n_features))
    target = np.empty((size-1,9))
    for i,ir in enumerate(data_file):
        del ir[0]
        del ir[0]
        del ir[0]
        del ir[0]
        del ir[1]
        del ir[2]
        del ir[25]
        del ir[24]
        if ir[0]=='':
            continue
        ir.append(0)
        ir.append(0)
        ir.append(0)
        

        if data_file.line_num ==1:
            data[i]=np.asarray(ir,dtype=np.float32)
            last_sell = float(data[i][7])
            last_buy = float(data[i][8])
        elif data_file.line_num == size:
            labellist=[]
            labellist = get_class(float(ir[7]),float(ir[8]),last_sell,last_buy)
            target[i-1]=np.asarray(labellist,dtype=np.float32)
        else:
            data[i]=np.asarray(ir,dtype=np.float32)
            labellist=[]
            labellist = get_class(float(data[i][7]),float(data[i][8]),last_sell,last_buy)
            target[i-1]=np.asarray(labellist,dtype=np.float32)
            last_sell = float(data[i][7])
            last_buy = float(data[i][8])
    for i in range(0,size-1):
        data[i][n_features-1] = avg_price(data,i,volume_index,totalprice_index,para)
        data[i][n_features-2] = MA_lastprice(data,i,lastprice_index,5)
        data[i][n_features-3] = MA_lastprice(data,i,lastprice_index,20)
        #print data[i]
        #raw_input("pause")

    data_scaled=preprocessing.scale(data)
    #target_scaled = preprocessing.scale(target)
    #print data[1]
    #print "-----------------------"
    #print target
    return Dataset(data=data_scaled,target=target)

def load_csv_data_withoutdeepdata_softmax(filename,size,n_features):
    data_file = csv.reader(open(filename,'rb'))
    data = np.empty((size-1,9))
    data_r = np.empty((size-1,n_features))
    target = np.empty((size-1,2))
    for i,ir in enumerate(data_file):
        del ir[0]
        del ir[0]
        del ir[0]
        del ir[0]
        del ir[1]
        del ir[2]
        del ir[3]
        del ir[3]
        del ir[3]
        del ir[3]
        for j in range(0,8):
            del ir[5]
        for j in range(0,7):
            del ir[7]
        if ir[0]=='':
            continue
        if data_file.line_num != size:
            ir.append(0)
            ir.append(0)
            data[i] = np.asarray(ir,dtype=np.float64)
            ind = len(ir)-2
            ir[ind]=avg_price(data,i,volume_index,totalprice_index,para)
            data[i] = np.asarray(ir,dtype=np.float64)
        else:
            break
    for i in range(0,size-1):
        data[i][8] = MA_avgprice(data,i,volume_index,totalprice_index,20,para)
        tem = data[i]
        il = []
        il.append(data[i][7])
        il.append(data[i][3])
        il.append(data[i][4])
        il.append(data[i][5])
        il.append(data[i][6])
        il.append(data[i][7]-data[i][8])
        labellist=[]
        labellist = get_class_commodity(data,i,sell_index,buy_index,size)
        target[i] = np.asarray(labellist,dtype=np.float64)
        data_r[i] = np.asarray(il,dtype=np.float64)
    data_scaled=preprocessing.scale(data) 
    return Dataset(data=data_scaled,target=target)
           

def next_batch_random(dataset,num_step,batch_size,size,struct_len,n_out):
    indexs = set()
    next_data = np.empty((batch_size,num_step*struct_len))
    next_label = np.empty((batch_size,n_out))
    for i in range(0,batch_size):
        while True:
            tem = random.randint(num_step,size-1)
            last_len = len(indexs)
            indexs.add(tem)
            if len(indexs) > last_len:
                break
    i = 0
    for index in indexs:
        tem_data = dataset.data[index-num_step:index]
        tem_label = dataset.target[index-1]
        tem_data = np.reshape(tem_data,-1)
        tem_label = np.reshape(tem_label,-1)
        next_data[i] = tem_data
        next_label[i] = tem_label
        i+=1
    #print indexs
    #print next_data
    #print '------------------------'
    #print next_label
    return next_data,next_label

def next_batch_list(dataset,num_step,batch_size,size,struct_len,n_out):
    global batch_index
    start_index = batch_index
    tem_index = batch_index
    '''
    if start_index + batch_size*num_step > size-1:
        end_index = size-1
    else:
        end_index = start_index + batch_size*num_step
    '''
    next_data = np.empty((batch_size,num_step*struct_len))
    next_label = np.empty((batch_size,n_out))
    for i in range(0,batch_size):
        if tem_index+num_step > size-1:
            tem_index = 0
        tem_data = dataset.data[tem_index:tem_index+num_step]
        tem_label = dataset.target[tem_index+num_step-1]
        tem_data = np.reshape(tem_data,-1)
        tem_label = np.reshape(tem_label,-1)

        next_data[i] = tem_data
        next_label[i] = tem_label
        tem_index += 1

    batch_index = tem_index
    return next_data,next_label




#train = load_csv_data_withoutdeepdata_softmax("M1609_train.csv",600000,6)
#train = load_csv_data_mse("M1609.csv",[12,13,14,15,16,17,18,19,20,21],29,29)
#train = load_csv_data_softmax("IF1607_train.csv",300000,27)
#next_batch(train,10,10,29,29)

