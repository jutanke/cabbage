import os
from os.path import isfile, join
current_loc = os.path.dirname(os.path.realpath(__file__))

# delete old settings
if isfile('settings.txt'):
    os.remove('settings.txt')

deepmatch = join(current_loc, 'deepmatching_bin/deepmatching_1.2.2_c++/deepmatching-static')
graph_solver = join(current_loc, 'build/sample')

with open('settings.txt', 'w') as f:
    f.write("{\n")
    f.write('\t"data_root": "/tmp/mpt",\n')
    f.write('\t"deepmatching":"' + deepmatch + '",\n')
    f.write('\t"graph_solver":"' + graph_solver + '"\n}')
