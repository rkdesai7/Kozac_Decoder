import argparse
import matplotlib
import random
import sys
import itertools
import gzip
import json

parser = argparse.ArgumentParser(description="Extracts kozac sequence data from 5' utr and cds .gz files from a specific organism")
parser.add_argument("cds", type=str, help="Path to cds .gz file")
parser.add_argument("utr", type=str, help="Path to 5' utr .gz file")
parser.add_argument("--upstream", type=int, default=5, help="How many base pairs you want before the ATG in the kozac sequence")
parser.add_argument("--downstream", type=int, default=5, help="How many base pairs you want after the ATG in the kozac sequence")
parser.add_argument("--size_real", type=int, default=1000, help="How many real kozac sequences you want")
parser.add_argument("--size_fake", type=int, default=1000, help="How many fake kozac sequences you want")

arg=parser.parse_args()

def readfasta(filename):
	"""Simple fasta file iterator."""
	print(type(filename))
	name = None
	seqs = []

	fp = None
	if   filename.endswith('.gz'): fp = gzip.open(filename, 'rt')
	elif filename == '-':          fp = sys.stdin
	else:                          fp = open(filename)

	while True:
		line = fp.readline()
		if line == '': break
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

def write_specified_size_to_file(data, file_name, size):
	"""get specified size of data and write to a text file"""
	
	data = random.sample(data, size)
	with open(file_name, 'w') as f:
		for i in data:
			f.write(i)
			
def gen_real(utr_data, cds_data, upstream, downstream, size_real):
	"""extracts real kozac sequences"""
	utr5 = {}
	cds = {}
	data = []
	
	for i in readfasta(utr_data):
		name = i[0].split()[0]
		w = -1*upstream
		sequence = i[1][w:]
		utr5[name] = sequence
	
	for i in readfasta(cds_data):
		name = i[0].split()[0]
		sequence = i[1][:3+downstream]
		if sequence[:3] == "ATG": cds[name] = sequence
		
	for key in utr5:
		if key in cds:
			if len(utr5[key] + cds[key]) != 3 + downstream + upstream: continue
			data.append(key + "|" + utr5[key] + cds[key] + "\n")
		
	write_specified_size_to_file(data, "real_kozac.txt", size_real)
	gen_all(data, 'all_kozac.txt')
	print(f"Real Kozac Data of Size {arg.size_real} Generated")
	
def gen_fake(cds, upstream, downstream, size_fake):
	"""compiles fake kozac sequences (ATG codons not part of promoter)"""
	data = []
	
	for i in readfasta(cds):
		name = i[0].split()[0]
		seq = i[1][3:]
		for j in range(len(seq) - 1):
			win = seq[j:j+3]
			if win == "ATG":
				kozac = seq[j-upstream:j+3+downstream]
				if len(kozac) != 3 + upstream + downstream: continue
				text = name + "|" + kozac + "\n"
				data.append(text)
	
	write_specified_size_to_file(data, "fake_kozac.txt", size_fake)
	print(f"Fake Kozac Data of Size {arg.size_fake} Generated")

def gen_all(data, file_name):
	"""compiles all ground truth kozacs"""
	with open(file_name, 'w') as f:
		for i in data:
			f.write(i)

gen_real(arg.utr, arg.cds, arg.upstream, arg.downstream, arg.size_real)
gen_fake(arg.cds, arg.upstream, arg.downstream, arg.size_fake)
		

				
	
