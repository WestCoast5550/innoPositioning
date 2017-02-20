import random


f = open('out.txt', 'w')
destX = 400
destY = 600
startX = 0
startY = 0
n = 800.0

#generates test data
for i in range(800):
	x = i * (destX - startX) / n
	y = i * (destY - startY) / n

	#adding noise
	noiseX = random.uniform(-0.1, 0.1)
	noiseY = random.uniform(-0.1, 0.1)
	f.write(repr(x + noiseX) + ' ' + repr(y + noiseY) + '\n')