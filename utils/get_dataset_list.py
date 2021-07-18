import os
import numpy as np

dataset_dir = 'data/cifar100/test'
output_file = 'data/cifar100/test.csv'

output = []
classes = sorted(os.listdir(dataset_dir))

for clas in classes:
    imgs = sorted(os.listdir(os.path.join(dataset_dir, clas)))
    for img in imgs:
        output.append([os.path.join(clas, img), 'labeled'])

output = np.array(output)
np.savetxt(output_file, output, fmt="%s,%s")
print("saving to: ",output_file)
