from pathlib import Path
from funcs import *

directory = Path('DATA/')


for file_path in directory.glob('*.mat'):
    file = str(file_path).split('\\')[1]
    if 'baseline' in file:
        ALL = True
        BPFI = False
        BPFO = False
    elif 'Inner' in file:
        ALL = False
        BPFI = True
        BPFO = False
    elif 'Outer' in file:
        ALL = False
        BPFI = False
        BPFO = True
    else:
        raise AttributeError('Invalid File Name')

    complete_analysis(file, BPFI=BPFI, BPFO=BPFO, ALL=ALL)

# complete_analysis('OuterRaceFault_2.mat', BPFO=True)

print('\n>> DONE!')


# plt.show()