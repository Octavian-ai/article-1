
import random
import numpy as np
import tensorflow as tf
import math
import string
import copy
import uuid

class GeneticParam(object):
  """Represents a parameter that can be sampled, copied, compared and mutated"""
  
  def __init__(self):
    """Initialise with randomly sampled value"""
    pass
  
  def mutate(self, heat):
    """Return copy of this param with mutated value"""
    pass
  
  def __eq__(self, other):
    return self.value == other.value
  
  def __str__(self):
    return str(self.value)
  
  @property
  def value(self):
    return self.v    



class FixedParam(GeneticParam):
  def __init__(self,v=None):
    self.v = v
    
  def mutate(self, heat):
    return self

  def __str__(self):
    return ""
    
def FixedParamOf(v):
  return lambda: FixedParam(v)


class InitableParam(GeneticParam):
    def __init__(self,v=None):
        self.v = v
    


class MulParam(InitableParam):
  def __init__(self, v, min, max):
    self.v = v
    self.max = max
    self.min = min

  @property
  def value(self):
    return min(max(self.v, self.min), self.max)

  def mutate(self, heat):
    return MulParam(self.value * random.uniform(0.8/heat, 1.2*heat) + random.gauss(0.0, heat*0.1), self.min, self.max)

def MulParamOf(v, min=-10000, max=10000):
  return lambda: MulParam(v, min, max)



class IntParam(MulParam):
    @property
    def value(self):
        return round(min(max(self.v, self.min), self.max))

def IntParamOf(v, min=1, max=1000):
  return lambda: IntParam(v, min, max)



class LRParam(GeneticParam):
  def __init__(self, v=None):
    sample = pow(10, random.uniform(-4, 2))
    self.v = v if v is not None else sample
    
  def mutate(self, heat):
    return LRParam(self.v * pow(10, heat*random.uniform(-0.5,0.5)))
  


class Heritage(GeneticParam):
    
    def vend(self):
        return random.choice(string.ascii_letters)
    
    def __init__(self, v=""):
        self.v = v + self.vend()
    
    def mutate(self, heat):
        return Heritage(self.v)



""" Gives a fresh model folder name every mutation """
class ModelId(GeneticParam):
  
  def vend(self):
      return str(uuid.uuid1())
  
  def __init__(self, v={}):
      self.v = {
        "cur": self.vend(), 
        "warm_start_from": v.get("cur", None)
      }
  
  def mutate(self, heat):
      return ModelId(self.v)
    


class VariableParam(InitableParam):
    
  def __eq__(self, other):
    
    if self.v is None or other.v is None:
      return False
    
    for i in zip(self.v, other.v):
      if not np.array_equal(i[0], i[1]):
        return False
    
    return True
  
  def mutate(self, heat):
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
    
  def mutate(self, heat):
    o = self.value
    
    if random.random() > 1 - 0.2*heat:
      o = random.choice(self.choices)
    
    return OptimizerParam(o)


