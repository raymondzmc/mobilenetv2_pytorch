layers = [
	('conv2d', -1, 32, 1, 2),
	('bottleneck', 1, 16, 1, 1),
	('bottleneck', 6, 24, 2, 2),
	('bottleneck', 6, 32, 3, 2),
	('bottleneck', 6, 64, 4, 2),
	('bottleneck', 6, 96, 3, 1),
	('bottleneck', 6, 160, 3, 2),
	('bottleneck', 6, 320, 1, 1),
	('conv2d 1x1', -1, 1280, 1, 1),
]

classes = 1000
input_size = 224
width_mult = 1.