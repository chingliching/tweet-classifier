''' filter for tweets written by that user '''
import sys
filename = sys.argv[-1]

username = filename[:-4]

import csv
reader = csv.reader(open(filename,encoding='utf8'),delimiter=';')
print('original length is',sum(1 for row in reader))

writer = open(username+'_f.csv','w', encoding="utf8")
reader = csv.reader(open(filename,encoding='utf8'),delimiter=';')

header= next(reader)
writer.write(str(';'.join(header)))
length = len(header)
# print(length)

for line in reader:
	# if line[0].lower()==username.lower() and len(line)==length:
	# print(len(line))
	if len(line)==length:
		writer.write(';'.join(line) + '\n')
writer.close()

reader = csv.reader(open(username+'_f.csv',encoding='utf8'),delimiter=';')
print('final length is',sum(1 for row in reader))
