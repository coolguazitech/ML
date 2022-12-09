import tensorflow as tf

from itertools import combinations
x=[1,2,3,4,5,6]
c = combinations(x, 2)
for i in c:
    print(i)