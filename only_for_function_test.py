import numpy as np
import torch

class A(object):
    def fun1(self):
        print('fun1')
    def fun2(self):
        print('fun2')
        self.fun1()


a = A()
a.fun1()
a.fun2()