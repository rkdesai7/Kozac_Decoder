import argparse
import math

parser = argparse.ArgumentParser(desrciption = "Finding Lambda")
parser.add_argument('-A', type=float, default=.25, help='Percentage of A')
parser.add_argument('-match', type=float, default=1.0, help="Match Score")
parser.add_argument('-mis', type=float, default=-1.0, help='Mismatch Score')
parser.add_argument('-prec', type=float, default=1e-6, help='Precision')

arg=parser.parse_args()

seq1 = 'ATGC'
seq2 = 'ATGC'
g_comp = (1-arg.A)/2

weights = {'A': arg.A, 'C': g_comp, 'G': g_comp, 'T': arg.A}
scores = {'match':arg.match, 'mismatch':arg.mis)


def qij(nt1, nt2, lamb, weights, scores):
	"""Return p_ip_js_ij for each pairwise alignment"""
	if nt1 == nt2: return weights[nt1] * weights[nt2] * math.exp(lamb*scores['match'])
	if nt1 != nt2: return weights[nt1] * weights[nt2] * math.exp(lmab*scores['mismatch'])

max_lamb = 5
min_lamb = 0
	
while max_lamb-min_lamb > arg.prec:
	lamb = (max_lamb - min_lamb)/2	
	qtot = 0
	for nt1 in seq1:
		for nt2 in seq2:
			qtot += qij(nt1, nt2, lamb, weights, scores)
			
	if qtot > 1: max_lamb = lamb
	else:        min_lamb = lamb
	print(lamb)
		
