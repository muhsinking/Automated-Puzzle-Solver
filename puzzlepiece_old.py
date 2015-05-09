class PuzzlePiece:

	def __init__(self, contour = [], pixels = [], edges = [], corners = [], angles = []):
		self.contour = contour
		self.pixels = pixels
		self.edges = edges
		self.corners = corners
		self.angles = angles

	def setContour(self, cont):
		self.contour = cont

	def setPixels(self, pix):
		self.pixels = pix

	def setEdges(self, ed):
		self.edges = ed

	def setCorners(self, cn):
		self.corners = cn

	def setAngles(self, an):
		self.angles = an 

	def getContour(self):
		return self.contour

	def getPixels(self):
		return self.pixels

	def getEdges(self):
		return self.edges

	def getCorners(self):
		return self.corners

	def getAngles(self):
		return self.angles