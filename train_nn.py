import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser(description = "Trains a neural network to identify the kozac sequence")
parser.add_argument("real_data", type=str, help="Path to data containing a sample real kozac sequences")
parser.add_argument("fake_data", type=str, help="Path to data contaiing fake kozac sequences")
parser.add_argument("ground_truth", type=str, help="Path to data containing all known kozac sequences")
parser.add_argument("--encoder", type=str, default="pwm", help="How you want to numerically encode the data (pwm, binary, one_hot)")
parser.add_argument("--train_proportion", type=float, default=.75, help="Percentage of data you want in the training set")
parser.add_argument("--units", type=int, default=64, help="Number of units in each hidden layer")
parser.add_argument("--activation",type=str, default="relu", help="Activation function")
parser.add_argument("--batches", type=int, default=50, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

arg = parser.parse_args()

def data_to_df(real_data, false_data):
	"""Compile real and fake data into pandas dataframe"""
	data = []
	#real
	with open(real_data, "r") as file:
		for line in file:
			elements = line.strip().split('|')
			seq = list(elements[1])
			seq.append(1)
			data.append(seq)
	#fake
	with open(false_data, "r") as file:
		for line in file:
			elements = line.strip().split('|')
			seq = list(elements[1])
			seq.append(0)
			data.append(seq)
	df = pd.DataFrame(data)
	return df
	

def one_hot_encode(real, fake):
	"""Perform one hot encoding"""
	data = data_to_df(real, fake)
	mappings = {'A': '1000', 'T': '0100', 'G': '0010', 'C': '0001'}
	data.columns = [f'pos{i+1}' for i in range(data.shape[1] - 1)] + ["y"]
	X = data.iloc[:, :-1].copy()
	y = data.iloc[:, -1].copy()
	encoded_df=X.map(lambda nt: mappings.get(nt, '0000'))
	bit_df = encoded_df.map(lambda bits: [int(b) for b in bits])
	expanded_df = pd.DataFrame({
		f'{col}_{i}': bit_df[col].apply(lambda x:x[i])
		for col in bit_df.columns
		for i in range(4)
	})
		
	data = pd.concat([expanded_df, y], axis=1)
	return data

def binary_encode(real, fake):
	"""Perform binary encoding"""
	data = data_to_df(real, fake)
	mappings = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
	data.columns = [f'pos{i+1}' for i in range(data.shape[1] - 1)] + ["y"]
	X = data.iloc[:, :-1].copy()
	y = data.iloc[:, -1].copy()
	encoded_df=X.map(lambda nt: mappings.get(nt, '00'))
	bit_df = encoded_df.map(lambda bits: [int(b) for b in bits])
	expanded_df = pd.DataFrame({
		f'{col}_{i}': bit_df[col].apply(lambda x:x[i])
		for col in bit_df.columns
		for i in range(2)
	})
		
	data = pd.concat([expanded_df, y], axis=1)
	return data

def pwm_encode(real, fake, ground_truth):
	"""Perform encoding using the observed probability of a base being in a particular position"""
	frequencies = gen_pwm(ground_truth)
	data = data_to_df(real, fake)
	col_num = 0
	X = data.iloc[:, :-1].copy()
	y = data.iloc[:, -1].copy()
	for col in X.columns:
		for index, value in enumerate(data[col]):
			if value == 'A': X.loc[index, col] = frequencies[col_num][0]
			if value == 'T': X.loc[index, col] = frequencies[col_num][1]
			if value == 'G': X.loc[index, col] = frequencies[col_num][2]
			if value == 'C': X.loc[index, col] = frequencies[col_num][3]
		col_num += 1
	data = pd.concat([X, y], axis=1)
	return data
		

def gen_pwm(full_data):
	"""Return probabilities of a base being in a particular position based on an entire kozac dataset"""
	sequences = []
	with open(full_data, "r") as file:
		for line in file:
			lines = line.strip().split('|')
			if lines[1][5:8] != "ATG": continue
			seq = list(lines[1])
			sequences.append(seq)
	#ATGC
	counts = []
	for i in range(len(seq)):
		counts.append([0, 0, 0, 0])
	for i in sequences:
		for index, item in enumerate(i):
			if item == 'A': counts[index][0] += 1
			if item == 'T': counts[index][1] += 1
			if item == 'G': counts[index][2] += 1
			if item == 'C': counts[index][3] += 1
	for ind, i in enumerate(counts): 
		total = sum(i)
		for index, item in enumerate(i):
			counts[ind][index] = item/total
	return counts

def train_test_split(df, percentage):
	"""Split data into training and validation"""
	train_df = df.sample(frac = percentage, random_state = 4)
	val_df = df.drop(train_df.index)
	
	print(np.isnan(train_df).any())
	X_train = tf.convert_to_tensor(train_df.iloc[:,:-1].astype(float))
	X_val = tf.convert_to_tensor(val_df.iloc[:,:-1].astype(float))
	y_train = tf.convert_to_tensor(train_df.iloc[:,-1].astype(float))
	y_val = tf.convert_to_tensor(val_df.iloc[:,-1].astype(float))

	input_shape = [X_train.shape[1]]
	
	return X_train, X_val, y_train, y_val, input_shape
	
def train_nn(real, fake, ground_truth):
	"""Train neural network"""
	if arg.encoder == "one_hot": data = one_hot_encode(real, fake)
	if arg.encoder == "binary": data = binary_encode(real, fake)
	if arg.encoder == "pwm": data = pwm_encode(real, fake, ground_truth)
	
	X_train, X_val, y_train, y_val, input_shape = train_test_split(data, arg.train_proportion)
	
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=arg.units, activation=arg.activation, input_shape=input_shape),
		tf.keras.layers.Dense(units=arg.units, activation=arg.activation),
		tf.keras.layers.Dense(units=1, activation="sigmoid")])
		
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	
	losses = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=arg.batches, epochs=arg.epochs)
	loss_df = pd.DataFrame(losses.history)
	ax = loss_df.loc[:,['loss','val_loss']].plot()
	ax.set_title(f"Training Metrics for {arg.encoder} encoding")
	
	return model


model = train_nn(arg.real_data, arg.fake_data, arg.ground_truth)
 
	
	
	
