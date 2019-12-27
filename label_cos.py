import os
import math
path_1 = '/Users/george/Documents/GitKraken/annotations_1'
path_2 = '/Users/george/Documents/GitKraken/annotations_2'
for filename in os.listdir(path_1):
    file = open(path_1 + '/' + filename, 'r')
    fout = open(path_2 + '/' + filename, 'w')
    for line in file:
        var = line.rstrip().split(' ')
        angle = float(var[4])
        angle = math.cos(angle)
        var[4] = str(angle)
        new_line = ' '.join(var)
        fout.write(new_line + '\n')
    file.close()
    fout.close()
