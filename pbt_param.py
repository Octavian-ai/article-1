
import random
import numpy as np
import tensorflow as tf
import math
import string
import copy

class GeneticParam(object):
  """Represents a parameter that can be sampled, copied, compared and mutated"""
  
  def __init__(self):
    """Initialise with randomly sampled value"""
    pass
  
  def mutate(self):
    """Return copy of this param with mutated value"""
    pass
  
  def __eq__(self, other):
    return self.v == other.v
  
  def __str__(self):
    return str(self.value)
  
  @property
  def value(self):
    return self.v    



class FixedParam(GeneticParam):
  def __init__(self,v=None):
    self.v = v
    
  def mutate(self):
    return self

  def __str__(self):
    return ""
    
def FixedParamOf(v):
  return lambda: FixedParam(v)


class InitableParam(GeneticParam):
    def __init__(self,v=None):
        self.v = v
    

    
class MulParam(InitableParam):
  def mutate(self):
    return MulParam(self.v * random.uniform(0.8, 1.2))

def MulParamOf(v):
  return lambda: MulParam(v)



class IntParam(InitableParam):
    def __init__(self, v=None, min=1, max=1000):
      self.v = v
      self.max = max
      self.min = min
  
    @property
    def value(self):
        return round(min(max(self.min, self.v), self.max))
    
    def mutate(self):
        return IntParam(self.v * random.uniform(0.8, 1.2), self.min, self.max)


class LRParam(GeneticParam):
  def __init__(self, v=None):
    sample = pow(10, random.uniform(-4, 2))
    self.v = v if v is not None else sample
    
  def mutate(self):
    return LRParam(self.v * pow(10, random.uniform(-0.5,0.5)))
  
class Heritage(GeneticParam):
    
    def vend(self):
        return random.choice(string.ascii_letters)
    
    def __init__(self, v=""):
        self.v = v + self.vend()
    
    def mutate(self):
        return Heritage(self.v)
    
class VariableParam(InitableParam):
    
  def __eq__(self, other):
    
    if self.v is None or other.v is None:
      return False
    
    for i in zip(self.v, other.v):
      if not np.array_equal(i[0], i[1]):
        return False
    
    return True
  
  def mutate(self):
    return VariableParam(copy.copy(self.v))
  
  def __str__(self):
    return ""
  

class OptimizerParam(GeneticParam):
  def __init__(self, v=None):
    self.choices = [
        tf.train.AdamOptimizer,
        tf.train.RMSPropOptimizer,
        tf.train.GradientDescentOptimizer,
        tf.train.AdagradOptimizer
    ]
    self.v = v if v is not None else random.choice(self.choices)
    
  def mutate(self):
    o = self.value
    
    if random.random() > 0.8:
      o = random.choice(self.choices)
    
    return OptimizerParam(o)


