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
map_s = {
    "X" : 1,
    "S" : 1,
    "-" : 0,
    "?" : 1,
    "Q" : 1,
    "E" : 1,
    "<" : 1,
    ">" : 1,
    "[" : 1,
    "]" : 1,
    "o" : 1,
    "B" : 1,
    "b" : 1
    }

def generate_conditions(level):
    ground_not_fully_tiled = 1 if 2 in level[n_x - 1] else 0
    pipes = 1 if 8 in level else 0
    enemies = 1 if 5 in level else 0
    pyramids = 1 if 0 in level[:n_x - 1] else 0
    floating_tiles = 1 if (1 in level) or (3 in level) or (4 in level) else 0
    print([ground_not_fully_tiled, pipes, enemies, pyramids, floating_tiles])
    return np.asarray([grsound_not_fully_tiled, pipes, enemies, pyramids, floating_tiles])

def generate_levels(index):
    level = None
    level_structure = None
    level_width = 0
    f = open('mario-' + str(index + 1) + '.txt', 'r')

    for i in range(0, n_x):
        line = f.readline()
        if i == 0:
            level = np.zeros((n_x, len(line.rstrip('\n'))))
            level_structure = np.zeros((n_x, len(line.rstrip('\n'))))
            level_width = len(line.rstrip('\n'))
        line_parsed = [map[symbol] for symbol in list(line.rstrip('\n'))]
        level[i] = line_parsed
        line_parsed = [map_s[symbol] for symbol in list(line.rstrip('\n'))]
        level_structure[i] = line_parsed

    levels = []
    levels_structure = []
    conditions = []
    for i in range(0, level_width - n_y + 1):
        level_segment = level[:, i:i+n_y]
        level_structure_segment = level_structure[:, i:i+n_y]
        levels.append(level_segment)
        levels_structure.append(level_structure_segment)
        condition = generate_conditions(level_segment)
        conditions.append(condition)

    print('Level ' + str(index + 1) + ':')
    print('Level Width: ' + str(level_width))
    print(np.asarray(levels).shape)
    print(np.asarray(levels_structure).shape)
    return levels, levels_structure, conditions

train = []
train_structure = []
train_condition = []
for i in range(0, n_levels):
    train_example = generate_levels(i)
    train.extend(train_example[0])
    train_structure.extend(train_example[1])
    train_condition.extend(train_example[2])
train = np.asarray(train).astype(int)
train_structure = np.asarray(train_structure).astype(int)
train_condition = np.asarray(train_condition).astype(int)

print('Train Set:')
print(train)
print(train.shape)

print('Train Structure Set:')
print(train_structure)
print(train_structure.shape)

print('Train Condition Set:')
print(train_condition)
print(train_condition.shape)

out_file = open("example_super_mario_bros_1.json", "w")
json.dump(train_structure.tolist(), out_file)
out_file.close()

out_file = open("example_super_mario_bros_2.json", "w")
json.dump(train.tolist(), out_file)
out_file.close()

out_file = open("example_super_mario_bros_2_cond.json", "w")
json.dump(train_condition.tolist(), out_file)
out_file.close()
