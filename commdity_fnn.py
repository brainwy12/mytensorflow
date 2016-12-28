import tensorflow as tf 
import numpy as np
import readcsv
from tensorflow.python.ops import rnn,rnn_cell
import csv
import math
#log write
log_data_csv = file("log_data.csv",'wb')
log_label_csv = file("log_file.csv",'wb')
data_writer = csv.writer(log_data_csv)
lable_writer = csv.writer(log_label_csv)
#parameters
learning_rate = 0.01
training_iters =15000
batch_size = 512
display_step = 100

#network parameters
n_input = 6
n_output = 2
n_time_step = 20
n_hidden_1 = 15
n_hidden_2 = 10
keep_prob = 0.8
#Graph input
x = tf.placeholder("float",[None,n_time_step,n_input])
y = tf.placeholder("float",[None,n_output])

#data parameter
train_size = 300000
test_size = 100000
valid_size = 29
trainfile = "M1609_train.csv"
validfile = "M1609.csv"
testfile = "M1609_test.csv"


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}

def RNN(x,is_training,weights,biases):
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(0,n_time_step,x)

    lstm_cell_1 = rnn_cell.LSTMCell(n_hidden_1,forget_bias=0.8)
    lstm_cell_2 = rnn_cell.LSTMCell(n_hidden_2,forget_bias=0.8)
    
    if is_training and keep_prob < 1:
        lstm_cell_1 = rnn_cell.DropoutWrapper(lstm_cell_1,output_keep_prob=keep_prob)
        lstm_cell_2 = rnn_cell.DropoutWrapper(lstm_cell_2,output_keep_prob=keep_prob)
    
    cell = rnn_cell.MultiRNNCell([lstm_cell_1,lstm_cell_2])
    
    #if is_training and keep_prob < 1:
    #    x = tf.nn.dropout(x,keep_prob)
    
    #initial_state = cell.zero_state(batch_size,tf.float32)
    #state = initial_state
    output = []
    output,states = rnn.rnn(cell,x,dtype = tf.float32)
    #outputs = tf.reshape(tf.concat(1,output),[-1,n_hidden_2])
    #maybe a softmax
    return tf.matmul(output[-1],weights['out'])+biases['out']



output_train = RNN(x,True,weights,biases)
#output_other = RNN(x,False,weights,biases)


#logits = tf.nn.softmax(output_train)
#define loss and optimizer
cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_train,y))

optimizer_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_train)


# initial variable
initial = tf.initialize_all_variables()

#read data
train = readcsv.load_csv_data_withoutdeepdata_softmax(trainfile,train_size,n_input)
#valid = readcsv.load_csv_data(validfile,[12,13,14,15,16,17,18,19,20,21],valid_size,29)
test = readcsv.load_csv_data_withoutdeepdata_softmax(testfile,test_size,n_input)

#eval model
correct_prediction = tf.equal(tf.argmax(output_train,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#run graph
with tf.Session() as sess:
    sess.run(initial)
    for i in range(training_iters):
        batch_x,batch_y = readcsv.next_batch_random(train,n_time_step,batch_size,train_size,n_input,n_output)
        #batch_x,batch_y = readcsv.next_batch_list(train,n_time_step,train_size-n_time_step,train_size,n_input,n_output)
        #print "get batch"
        batch_x = batch_x.reshape((batch_size,n_time_step,n_input))
        batch_y = batch_y.reshape((batch_size,n_output))
        #batch_x = batch_x.reshape((train_size-n_time_step,n_time_step,n_input))
        #batch_y = batch_y.reshape((train_size-n_time_step,n_output))
        #print batch_x.shape
        #print sess.run(output_train,feed_dict={x:batch_x,y:batch_y}).shape
        sess.run(optimizer_train,feed_dict={x:batch_x,y:batch_y})
        #print "weight is ------"
        #w=sess.run(weights)
        '''
        cost = sess.run(cost_train,feed_dict={x:batch_x,y:batch_y})
        if math.isnan(cost):
            print "there is wrong data in %d iter-------------------------" %i
            for j in range(train_size-n_time_step):
                for k in range(n_time_step):
                    for l in range(n_input):
                        if math.isnan(batch_x[j][k][l]):
                            print "there is a nan in x"
            print batch_x
            data_writer.writerows(batch_x)
            raw_input("111:")
        '''
        
        if i % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost_train,feed_dict={x:batch_x,y:batch_y})
            print "step %d , the accuracy is %g and the loss is %g" %(i,acc,loss)
    print "train finished"

    #test or valide
    batch_size_test = test_size-n_time_step
    test_x,test_y = readcsv.next_batch_list(test,n_time_step,batch_size_test,test_size,n_input,n_output)
    test_x = test_x.reshape((batch_size_test,n_time_step,n_input))
    test_y = test_y.reshape((batch_size_test,n_output))
    test_acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
    print "test accuracy is ", test_acc
    

