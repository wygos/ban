import numpy as np
from itertools import product

examples = {}

####################################################################
## This is the definition of the model
def f0(x):
    return x[0]


data0 = [[0], [1]]

examples[0] = (f0, data0)

####################################################################
## This is the definition of the model
def f1(x):
    if x[0] == 0 or x[1] == 0:
        return 1
    else:
        return 0


data1 = list(product(range(0,2), repeat=2))

examples[1] = (f1, data1)

####################################################################
def f2(x):
    if x[0] == 0:
        return 1
    elif x[1] == 0 or x[1] == 1:
        return 1
    else:
        return 0


data2 = list(product(range(0,2), range(0,4)))
examples[2] = (f2, data2)


####################################################################
def f3(x):
    if x[0] == 0 or (x[1] == 0 and x[2] == 0):
        return 1
    else:
        return 0



data3 = list(product(range(0,2), repeat=3))
examples[3] = (f3, data3)

####################################################################
def f4(x):
    if (x[0] == 0 and x[1] == 0) or (x[2] == 0 and (x[3] == 0 or x[4] == 0)):
        return 1
    else:
        return 0



data4 = list(product(range(0,2), repeat=5))
examples[4] = (f4, data4)

####################################################################
def f5(x):
    if (x[0] == 0 and x[1] == 0) or (x[0] == 0 and x[2] == 0)  or (x[1] == 0 and x[2] == 0)  or (x[0] == 0 and x[3] == 0):
        return 1
    else:
        return 0



data5 = list(product(range(0,2), repeat=4))
examples[5] = (f5, data5)

####################################################################

def f6(x):
    if (x[0] == 0 and x[1] == 1 and x[2] == 1) or (x[0] ==1 and x[1] == 0 and x[2]==1) or (x[0] == 1 and x[1]==1 and x[2]==0) :
        return 1
    else:
        return 0

data6 = list(product(range(0,2), repeat=3))
examples[6] = (f6, data6)



## takes f and data,
## return augmented version of f which is valid for nan's in input
def ex(f, data):
    data = np.array(data)
    n= len(data[0])
    def f_augmented(x):
        x = np.array(x)
        not_nan_ids = np.arange(n)[~np.isnan(x)]

        m = 0
        s = 0.
        for point in data:
            if np.all(point[not_nan_ids] == x[not_nan_ids]):
                #print(point, f(point))

                m += 1
                s += f(point)

        return s/m


    return f_augmented

examples = { k:(ex(f,d), d)  for k,(f,d) in examples.items()}


