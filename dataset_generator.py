import random
import sys
from setup.lib.korflab import readfasta

#Real
utr5 = {}
cds = {}
data = []
window = int(sys.argv[3])
for i in readfasta(sys.argv[1]):
	name = i[0].split()[0]
	w = -1*window
	sequence = i[1][w:]
	utr5[name] = sequence
for i in readfasta(sys.argv[2]):
	name = i[0].split()[0]
	sequence = i[1][:3+window]
	if sequence[:3] == "ATG": cds[name] = sequence
for key, value in utr5.items():
	if key in list(cds.keys()): data.append(key + "|" + utr5[key] + cds[key] + "\n")
for i in data:
	if i[5:8] != "ATG": data.remove(i)
data = random.sample(data, 1000)
with open('real_kozac.txt', 'w') as f:
	counter = 0
	for i in data:
		f.write(i)
print(counter)

#Fake ATG from coding region
false_data = []
for i in readfasta(sys.argv[2]):
	name = i[0].split()[0]
	seq = i[1][3:]
	for j in range(len(seq) - 4):
		win = seq[j:j + 3]
		if win == "ATG":
			kozac = seq[j-window:j+3+window]
			text = name + "|" + kozac + "\n"
			false_data.append(text)
false_data = random.sample(false_data, 1000)
with open('fake_kozac.txt', 'w') as f:
	for i in false_data:
		f.write(i)
		counter += 1
print(counter)
		

				
	
