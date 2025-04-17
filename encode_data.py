import pandas as pd

def extract_data(real_data, false_data):
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
	data = extract_data(real, fake)
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
	data = extract_data(real, fake)
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

def probability_encode(real, fake, ground_truth):
	"""Perform encoding using the observed probability of a base being in a particular position"""
	frequencies = get_probabilities(ground_truth)
	data = extract_data(real, fake)
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
		

def get_probabilities(full_data):
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
