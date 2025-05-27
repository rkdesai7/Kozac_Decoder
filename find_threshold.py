import argparse
import numpy as np

from scipy.optimize import minimize_scalar

parser = argparse.ArgumentParser(description = "Theshold optimizer intended for use with a kozac sequence")
parser.add_argument("real_data", type=str, help="Path to data containing a sample real kozac sequences")
parser.add_argument("fake_data", type=str, help="Path to data contaiing fake kozac sequences")
parser.add_argument("ground_truth", type=str, help="Path to data containing all known kozac sequences")
arg = parser.parse_args()

def gen_pwm(full_data):
	"""Return probabilities of a base being in a particular position based on an entire kozac dataset"""
	sequences = []
	with open(full_data, "r") as file:
		for line in file:
			lines = line.strip().split('|')
			seq = list(lines[1])
			sequences.append(seq)
	#ATGC
	blocks = len(sequences)//10
	
	beg = 0
	tot = []
	for i in range(len(seq)):
		tot.append([0, 0, 0, 0])
	counter = 0
	tot = np.array(tot)
	for i in range(10):
		end = beg + blocks
		sequence = sequences[:beg] + sequences[end+1:]
		counts = []
		for i in range(len(seq)):
			counts.append([0, 0, 0, 0])
		for i in sequence:
			for index, item in enumerate(i):
				if item == 'A': counts[index][0] += 1
				if item == 'T': counts[index][1] += 1
				if item == 'G': counts[index][2] += 1
				if item == 'C': counts[index][3] += 1
		for ind, i in enumerate(counts): 
			total = sum(i)
			for index, item in enumerate(i):
				if total == 0: counts[ind][index] = 0
				else:          counts[ind][index] = item/total
		tot = tot + np.array(counts)
		beg = end
		
	pwm = tot/10
	return pwm
	
	
def get_data(real, fake):
	
	data = []
	with open(real, "r") as file:
		for line in file:
			line = line.strip().split('|')
			data.append([line[1], 1])
	with open(fake, "r") as file:
		for line in file:
			line = line.strip().split('|')
			data.append([line[1], 0])
		
	return data
	
def find_prob(sequence, pwm):

	keys = {"A": 0, "T": 1, "G": 2, "C": 3}
	prob = 1
	for x, base in enumerate(sequence):
		index = keys[base]
		prob *= pwm[x][index]

	return prob
	
def optimize_thresh(data, pwm, left, right, depth=0, max_depth=20, min_width = 1e-10):
	def calc_accuracy(thresh):
		metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
		for i in data:
			seq = i[0]
			score = find_prob(seq, pwm)
			#print(score)
			if (i[1] == 0) and (score > thresh): metrics['FP'] += 1
			if (i[1] == 0) and (score < thresh): metrics['TN'] += 1
			if (i[1] == 1) and (score > thresh): metrics['TP'] += 1
			if (i[1] == 1) and (score < thresh): metrics['FN'] += 1
			
		accuracy = (metrics['TN'] + metrics['TP'])/(metrics['FN'] + metrics['FP'] + metrics['TN'] + metrics['TP'])
		return -1*accuracy
	if depth >= max_depth or (right-left) < min_width:
		mid = (left + right)/2
		return mid, calc_accuracy(mid)
	mid = (left + right)/2
	lmid = (left+mid)/2
	rmid = (mid+right)/2
	
	f_lmid = calc_accuracy(lmid)
	f_mid = calc_accuracy(mid)
	f_rmid = calc_accuracy(rmid)
	
	if f_lmid > f_mid: return optimize_thresh(data, pwm, left, mid, depth+1, max_depth, min_width)
	elif f_rmid > f_mid: return optimize_thresh(data, pwm, mid, right, depth+1, max_depth, min_width)
	else: return optimize_thresh(data, pwm, lmid, rmid, depth+1, max_depth, min_width)
	
#max = 0.00007
#min = 0.00006
#thresh_vals = []
pwm = gen_pwm(arg.ground_truth)
data = get_data(arg.real_data, arg.fake_data)
print(optimize_thresh(data, pwm, 0.0, .05))
#best_acc = 0
#best_thresh = None
#while max - min > 1e-15:
#	mid = (min + max)/2
#	acc = calc_accuracy(data, pwm, mid)
#	if acc > best_acc:
#		best_thresh = mid
#		best_acc = acc
#	min = mid
#print(f"Optimal threshold: {best_thresh}, Best Accuracy: {best_acc}")

	
