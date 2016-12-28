
def avg_price(data,index,volume_index,price_index,para):
	if data[index][volume_index] != 0:
		return data[index][price_index]/(data[index][volume_index]*para)
	else:
		return data[index][0]

def MA_lastprice(data,index,lastprice_index,dur):
	sum = 0.0
	if index - dur < -1:
		dur = index+1
	for i in range(0,dur):
		sum+=data[index-dur+i+1][lastprice_index]
	return sum/dur

def MA_avgprice(data,index,volume_index,price_index,dur,para):
	sum =0.0
	if index - dur < -1:
		dur = index+1
	for i in range(0,dur):
		sum+=avg_price(data,index-dur+1+i,volume_index,price_index,para)
	return sum/dur

def EMA_lastprice(data,index,lastprice_index,dur):
	if index - dur < -1:
		dur = index+1
	yesterday=data[index-dur+1][lastprice_index]
	ema_para = 2/(dur+1)
	for i in range(0,dur-1):
		today = (1-ema_para)*yesterday+ema_para*data[index-dur+i+2][lastprice_index]
		yesterday = today
	return today

def EMA_avgprice(data,index,volume_index,price_index,dur,para):
	if index - dur < -1:
		dur = index +1
	yesterday = avg_price(data,index-dur+1,volume_index,price_index,para)
	ema_para = 2/(dur+1)
	for i in range(0,dur-1):
		today = (1-ema_para)*yesterday+ema_para*avg_price(data,index-dur+i+2,volume_index,price_index,para)
		yesterday = today
	return today
	
def DIF_lastprice(data,index,lastprice_index):
	return EMA_lastprice(data,index,lastprice_index,12)-EMA_lastprice(data,index,lastprice_index,26)

def DEA_lastprice(data,index,lastprice_index,dur):
	if index-dur<-1:
		dur = index+1
	yesterday = DIF_lastprice(data,index-dur+1,lastprice_index)
	para = 2/(dur+1)
	for i in range(0,dur-1):
		today = (1-para)*yesterday+para*DIF_lastprice(data,index-dur+i+2)
		yesterday = today
	return today


def MACD_lastprice(data,index,lastprice_index):
	dif = DIF_lastprice(data,index,lastprice_index)
	dea = DEA_lastprice(data,index,lastprice_index)
	return 2*(dif-dea)
