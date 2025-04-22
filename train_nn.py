import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description = "Trains a neural network to identify the kozac sequence")
parser.add_argument("real_data", type=str, help="Path to data containing a sample real kozac sequences")
parser.add_argument("fake_data", type=str, help="Path to data contaiing fake kozac sequences")
parser.add_argument("ground_truth", type=str, help="Path to data containing all known kozac sequences")
parser.add_argument("--encoder", type=str, default="pwm", help="How you want to numerically encode the data (pwm, binary, one_hot)")
parser.add_argument("--train_proportion", type=float, default=.75, help="Percentage of data you want in the training set")
parser.add_argument("--units", type=int, default=30, help="Number of units in each hidden layer")
parser.add_argument("--activation",type=str, default="relu", help="Activation function")
parser.add_argument("--batches", type=int, default=50, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--k", type=int, default=10, help="Number of folds for cross validation")

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
	data = data.dropna()
	
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
	data = data.dropna()
	
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
	data = data.dropna()
	
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
	
def plot_training_metrics(metric_data):
	"""Plots loss and validation loss and saves graph"""
	metric_data = pd.DataFrame(metric_data.history)
	ax = metric_data.loc[:,['loss', 'val_loss']].plot()
	ax.set_title("Training Metrics for {arg.encoder} encoding")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	plt.tight_layout()
	plt.show()
	plt.savefig(f"training_metrics_{arg.encoder}.png")
	
def evaluate_model(model, X_val, y_val, fold):
	"""Calculate accuracy, variance, and bias to evaluate model in each fold"""
	predictions = model.predict(X_val).flatten()
	y_pred = (predictions > 0.5).astype(int)
	acc = accuracy_score(y_val, y_pred)
	bias = np.means((predictions-y_val)**2)
	variance = np.var(predictions)
	
	print(f"Fold {fold} metrics: accuracy: {acc:.4f} | bias: {bias:.4f} | variance: {variance:.4f}")
	
	return acc, bias, variance
	
def train_nn(real, fake, ground_truth):
	"""Train neural network"""
	#specify encoder
	if arg.encoder == "one_hot": data = one_hot_encode(real, fake)
	if arg.encoder == "binary": data = binary_encode(real, fake)
	if arg.encoder == "pwm": data = pwm_encode(real, fake, ground_truth)
	
	#prepare
	data = shuffle(data, random_state=42)
	X = data.iloc[:, :-1].astype(float).to_numpy()
	y = data.iloc[:, -1].astype(float).to_numpy()
	kf = KFold(n_split=arg.k, shuffle=True, random_state=42)
	
	#metrics to track
	best_accuracy = 0
	best_model = None
	best_losses = None
	best_bias = 0
	best_var = 0
	fold = 1
	
	#k-fold cross val
	for train_index, val_index in kf.split(X):
		print(f"\n Fold: {fold}/{k}")
		#train_test split
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]
		input_shape = [X_train.shape[1]]
	
		#model schematics
		model = tf.keras.Sequential([
			tf.keras.layers.Dense(units=arg.units, activation=arg.activation, input_shape=input_shape),
			tf.keras.layers.Dense(units=arg.units, activation=arg.activation),
			tf.keras.layers.Dense(units=arg.units, activation=arg.activation),
			tf.keras.layers.Dense(units=arg.units, activation=arg.activation),
			tf.keras.layers.Dense(units=1, activation="sigmoid")])
		
		#compile and train
		model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
		losses = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=arg.batches, epochs=arg.epochs, verbose = 0)
		acc, bias, variance = evaluate_model(model, X_val, y_val, fold)
		
		#update best metrics
		if acc > best_accuracy:
			best_accuracy = acc
			best_model = model
			best_losses = losses
			best_bias = bias
			best_var = variance
		
		fold += 1
		
	#final assesment and outputs
	print(f"\n Best model metrics across {arg.k} folds:\n accuracy: {best_accuracy:.4f}\nbias: {best_bias:.4f}\nvariance: {best_var:.4f}")
	plot_training_metrics(losses)
	best_model.save(f"model_{arg.encoder}_encoded.h5")
	
	return best_model

model = train_nn(arg.real_data, arg.fake_data, arg.ground_truth)
 
	
	
	
