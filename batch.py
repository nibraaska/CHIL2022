import os
import sys
import pathlib

for t_sub in [ "S" + str(i) for i in range(2, 18) if i != 12]:
    os.system('start cmd /k "conda activate tf-cpu && python {0} {1} && exit"'.format(sys.argv[1], t_sub))