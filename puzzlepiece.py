class PuzzlePiece:

	def __init__(self, uniqueID = -1, contour = [], pixels = [], edges = [], edgeContours = []):
		self.uniqueID = uniqueID
		self.contour = contour
		self.pixels = pixels
		self.edges = edges
		self.edgeContours = edgeContours

	def setID (self, uID):
		self.uniqueID = uID

	def setContour(self, cont):
		self.contour = cont

	def setPixels(self, pix):
		self.pixels = pix

	def setEdges(self, ed):
		self.edges = ed

	def setEdgeContours(self, edC):
		self.edgeContours = edC

	def getID(self):
		return self.uniqueID

	def getContour(self):
		return self.contour

	def getPixels(self):
		return self.pixels

	def getEdges(self):
		return self.edges

	def getEdgeContours(self):
		return self.edgeContours

	def getDimensions(self):
		shape = self.pixels.shape
		return (shape[0], shape[1])