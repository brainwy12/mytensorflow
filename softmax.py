import tensorflow as tf 
import numpy as np
import readcsv
import csv
import math
#log write
log_data_csv = file("log_data.csv",'wb')
log_label_csv = file("log_file.csv",'wb')
data_writer = csv.writer(log_data_csv)
lable_writer = csv.writer(log_label_csv)
#parameters
learning_rate = 0.01
train_iters = 10000
batch_size = 1024
display_step = 100

#network parameters
n_input = 9
n_output = 2
n_time_step = 1
#graph input
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_output])

#data parameter
train_size = 300000
test_size = 100000
trainfile = "M1609_train.csv"
testfile = "M1609_test.csv"



weights = tf.Variable(tf.zeros([n_input,n_output]))
bias = tf.Variable(tf.zeros([n_output]))

#write softmax
logits = tf.matmul(x,weights)+bias


#train
corss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y))
optimizer_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(corss_entropy)

#eval data
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#initial 
init = tf.initialize_all_variables()

#read data
train = readcsv.load_csv_data_withoutdeepdata_softmax(trainfile,train_size,6)
test = readcsv.load_csv_data_withoutdeepdata_softmax(testfile,test_size,6)

#run graph
with tf.Session() as sess:
	sess.run(init)
	for i in range(0,train_iters):
		batch_x,batch_y = readcsv.next_batch_random(train,n_time_step,batch_size,train_size,n_input,n_output)
		#batch_x,batch_y = readcsv.next_batch_list(train,n_time_step,train_size-n_time_step,train_size,n_input,n_output)
		sess.run(optimizer_train,feed_dict={x:batch_x,y:batch_y})
		
        	if i % display_step == 0:
        		acc = sess.run(accuarcy,feed_dict={x:batch_x,y:batch_y})
        		loss = sess.run(corss_entropy,feed_dict={x:batch_x,y:batch_y})
        		print "step %d , the accuracy is %g and the loss is %g" %(i,acc,loss)

	print "train finished"

	test_x,test_y = readcsv.next_batch_list(test,n_time_step,test_size-n_time_step,test_size,n_input,n_output)
	test_acc = sess.run(accuarcy,feed_dict={x:test_x,y:test_y})
	print "test accuarcy is ", test_acc

