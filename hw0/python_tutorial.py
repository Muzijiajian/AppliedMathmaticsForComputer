#encoding=utf-8

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [ x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print quicksort([3, 8, 6, 10, 1, 2, 1])

hello = "hello"
world = "world"
# sprintf style string formatting
hw12 = '%s %s %d' %(hello, world, 12)
print hw12

# If you want access to the index of each element within the body of a loop, use the built-in enumerate function
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)

# in the [] represents it is still list
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums if x % 2 == 0]
print squares
#in the {} represents it is dict
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square

d = {(x, x + 1): x for x in range(10)}
t = (5, 6)
print type(t)
print d[t]

class Greeter(object):
    def __init__(self, name):
        self.name = name
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name
g = Greeter('Fred')
g.greet()
g.greet(loud=True)

import numpy as np
e = np.random.random((2,2))
print e
