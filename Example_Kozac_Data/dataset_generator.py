import argparse
import matplotlib
import random
import sys
from setup.lib.korflab import readfasta

parser = argparse.ArgumentParser(description="Extracts kozac sequence data from 5' utr and cds .gz files from a specific organism")
parser.add_argument("cds", type=str, help="Path to cds .gz file")
parser.add_argument("utr", type=str, help="Path to 5' utr .gz file")
parser.add_argument("--window", type=int, default=5, help="How many base pairs you want before and after the ATG in the kozac sequence")
parser.add_argument("--size_real", type=int, default=1000, help="How many real kozac sequences you want")
parser.add_argument("--size_fake", type=int, default=1000, help="How many fake kozac sequences you want")

arg=parser.parse_args()

def write_specified_size_to_file(data, file_name, size):
	"""get specified size of data and write to a text file"""
	for index, i in enumerate(data):
		if i[arg.window:arg.window+3] != "ATG": del data[index]
		#if len(i) != 3+(2*arg.window): data.remove(i)
	data = random.sample(data, size)
	with open(file_name, 'w') as f:
		for i in data:
			f.write(i)
			
def gen_real():
	"""extracts real kozac sequences"""
	utr5 = {}
	cds = {}
	data = []
	
	for i in readfasta(arg.utr):
		name = i[0].split()[0]
		w = -1*arg.window
		sequence = i[1][w:]
		utr5[name] = sequence
	
	for i in readfasta(arg.cds):
		name = i[0].split()[0]
		sequence = i[1][:3+5]
		if sequence[:3] == "ATG": cds[name] = sequence
		
	for key, value in utr5.items():
		if key in cds:
			if len(utr5[key] + cds[key]) != 13: continue
			data.append(key + "|" + utr5[key] + cds[key] + "\n")
		
	write_specified_size_to_file(data, "real_kozac.txt", arg.size_real)
	gen_all(data, 'all_kozac.txt')
	print(f"Real Kozac Data of Size {arg.size_real} Generated")
	
def gen_fake():
	"""compiles fake kozac sequences (ATG codons not part of promoter)"""
	data = []
	
	for i in readfasta(arg.cds):
		print("reading")
		name = i[0].split()[0]
		seq = i[1][3:]
		for j in range(len(seq) - 1):
			win = seq[j:j+3]
			if win == "ATG":
				kozac = seq[j-arg.window:j+3+arg.window]
				if len(kozac) != 13: continue
				text = name + "|" + kozac + "\n"
				data.append(text)
	
	write_specified_size_to_file(data, "fake_kozac.txt", arg.size_fake)
	print(f"Fake Kozac Data of Size {arg.size_fake} Generated")

def gen_all(data, file_name):
	"""compiles all ground truth kozacs"""
	with open(file_name, 'w') as f:
		for i in data:
			f.write(i)

gen_real()
gen_fake()
		

				
	
