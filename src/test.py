import numpy

a = [0,1,2]
b = [0.1, 0.7, 0.2]
c = [0, 0, 0]

for i in range(20):
    c[numpy.random.choice(a=a, p=b)] += 1
print(c)
d = numpy.sum(a)

b = [a[i] / d for i in range(len(a))]
print(b)

c = [0, 0, 0]

for i in range(20):
    c[numpy.random.choice(p=b)] += 1
print(c)