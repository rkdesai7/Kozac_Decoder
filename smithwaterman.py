##Smith Waterman
def display_matrix(m, s1, s2):
	print('    ', end='')
	for i in range(len(s2)):
		print(s2[i], end='  ')
	print()
	
	for i in range(len(m)):
		if i == 0: print('  ', end='')
		if i != 0: print(s1[i-1], end=' ')
		for j in range(len(m[i])):
			print(m[i][j], end=' ')
		print()



s1 = 'ACGT' #vertical sequence
s2 = 'ACGGT' #horizontal sequence


score = []
for i in range(len(s1)+1):
	score.append(['.'] * (len(s2)+1))
	
for i in range(len(s1)+1): score[i][0] = 0
for j in range(len(s2)+1): score[0][j] = 0
display_matrix(score, s1, s2)

	score.append([])
	
