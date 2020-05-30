import sys
sys.path.insert(0, '../vision')
from torchvision.datasets import ImageFolder
from pdb import set_trace

__all__ = ['nyu_depth']

class NYUDepth():
	def __init__(self):
		set_trace()
		self.rgb = ImageFolder("datasets/nyu_depth_v2")
		pass

def nyu_depth():
	return NYUDepth()