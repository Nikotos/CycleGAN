"""
	Class to track parameters changing during training
"""

class InfoTracker:
	def __init__(self, name, averagingNumber):	
		self.name = name
		self.storage = []
		self.size = averagingNumber
		self.currentPos = 0

	def add(self, element)
		if len(self.storage) < self.size:
			self.storage.append(element)		
		else:
			self.storage[self.currentPos] = element
			self.currentPos = (self.currentPos + 1) % self.size


	def d(self, data):
		return "[" + str(data) + "]"

	def verbose(self, epoch, iter):
		mean = sum(self.storage) / self.size
		print("epoch -", d(epoch), "iter -", d(iter), name, "-", d(mean))