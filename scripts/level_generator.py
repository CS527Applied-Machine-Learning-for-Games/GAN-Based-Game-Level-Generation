import numpy as np

import json
from json import JSONEncoder

n_levels = 15
n_x = 14
n_y = 28
map = {
    "X" : 0,
    "S" : 1,
    "-" : 2,
    "?" : 3,
    "Q" : 4,
    "E" : 5,
    "<" : 6,
    ">" : 7,
    "[" : 8,
    "]" : 9,
    "o" : 10,
    "B" : 11,
    "b" : 12
    }

def generate_levels(index):
    level = None
    level_width = 0
    f = open('mario-' + str(index + 1) + '.txt', 'r')

    for i in range(0, n_x):
        line = f.readline()
        if i == 0:
            level = np.zeros((n_x, len(line.rstrip('\n'))))
            level_width = len(line.rstrip('\n'))
        line_parsed = [map[symbol] for symbol in list(line.rstrip('\n'))]
        level[i] = line_parsed

    levels = []
    for i in range(0, level_width - n_y + 1):
        levels.append(level[:, i:i+n_y])

    print('Level ' + str(index + 1) + ':')
    print('Level Width: ' + str(level_width))
    print(np.asarray(levels).shape)
    return levels

train = []
for i in range(0, n_levels):
    train.extend(generate_levels(i))
train = np.asarray(train)
train = train.astype(int)

print('Train Set:')
print(train)
print(train.shape)

out_file = open("example_super_mario_bros.json", "w")
json.dump(train.tolist(), out_file)
out_file.close()
