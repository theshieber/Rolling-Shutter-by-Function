import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random as r
import cv2
from PIL import Image
import scipy.misc
import ffmpeg
import subprocess as sp
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from os.path import isfile, join
import shutil

globalHeight = 0
globalWidth = 0

def init():
	os.mkdir('/Users/elishieber/Desktop/Thanksgiving/inframes')
	os.mkdir('/Users/elishieber/Desktop/Thanksgiving/outframes')

def cleanUp():
	try:
		shutil.rmtree('/Users/elishieber/Desktop/Thanksgiving/inframes')
	except:
		print("inframes not made yet")

	try:	
		shutil.rmtree('/Users/elishieber/Desktop/Thanksgiving/outframes')
	except:
		print("outframes not made yet")

def readMovie():
	os.chdir("/Users/elishieber/Desktop/Thanksgiving/inmovies")

	vidcap = cv2.VideoCapture('sunsetwalk.mov')
	success,image = vidcap.read()
	count = 0 

	os.chdir("/Users/elishieber/Desktop/Thanksgiving/inframes")

	while success:
	  cv2.imwrite("frame%.4d.jpg" % count, image)     # save frame as JPEG file      
	  success,image = vidcap.read()
	  count += 1

'''def rollingShutter():

	pathIn = '/Users/elishieber/Desktop/Thanksgiving/inframes'
	pathOut = '/Users/elishieber/Desktop/Thanksgiving/outframes'


	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	files.sort(key = lambda x: x[5:-4])
	print len(files)
	numFrames = len(files)
	Buffer = [[] for i in range(numFrames) ]

	globalPreImg = mpimg.imread(files[0])
	globalShape = globalPreImg.shape
	maxDelay = calcMaxDelay(globalShape[0], globalShape[1])
	print("calcMaxDelay:"  + str(calcMaxDelay(globalShape[0], globalShape[1])))
	curFrame = 0
	for filename in files:
		if (filename != ".DS_Store"):
			print curFrame
			print maxDelay
			if (curFrame > maxDelay):
				print filename
				os.chdir(pathIn)
				preImg = mpimg.imread(filename)
				Buffer[curFrame] = preImg
				shape = preImg.shape

				postImg = np.zeros(shape=shape)
				os.chdir(pathOut)
				for x in range(0, shape[0]):
					for y in range(0, shape[1]):

						delay = calcDelay(x, y)
						prevDelayFrame = int(np.floor(delay))
						nextDelayFrame = int(np.ceil(delay))
						#print delay
						#print (delay, prevDelayFrame, nextDelayFrame)
						#postImg[x][y] = Buffer[curFrame-delay][x][y]
						
						if(curFrame < delay):
							#postImg[x][y] = Buffer[curFrame][x][y]
							postImg[x][y] = [0,0,0]
						else:
							pixelOut = pixelWeightedAvg(Buffer[curFrame - prevDelayFrame][x][y], Buffer[curFrame - nextDelayFrame][x][y], delay-prevDelayFrame)
							postImg[x][y] = pixelOut
							#postImg[x][y] = [0,0,0]
						
				scipy.misc.imsave('outframes' + filename[-7:-4]  + '.jpg', postImg)
			curFrame += 1'''

def rollingShutter():

	pathIn = '/Users/elishieber/Desktop/Thanksgiving/inframes'
	pathOut = '/Users/elishieber/Desktop/Thanksgiving/outframes'


	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	#print files
	files.sort(key = lambda x: x[5:-4])
	print len(files)
	numFrames = len(files)

	globalPreImg = mpimg.imread(files[0])
	globalShape = globalPreImg.shape
	globalHeight = globalShape[0]
	globalWidth = globalShape[1]

	#Buffer = np.array([np.empty([globalHeight, globalWidth, 3]) for i in range(numFrames)])
	Buffer = np.empty([numFrames, globalHeight, globalWidth, 3])

	#maxDelay = calcMaxDelay(globalShape[0], globalShape[1])
	#print("calcMaxDelay:"  + str(calcMaxDelay(globalShape[0], globalShape[1])))
	curFrame = 0
	for filename in files:
		if (filename != ".DS_Store"):
			print filename
			os.chdir(pathIn)
			preImg = mpimg.imread(filename)
			#print preImg
			Buffer[curFrame] = np.array(preImg)

			postImg = np.zeros(shape=globalShape)
			os.chdir(pathOut)
			for x in range(0, globalShape[0]):
				for y in range(0, globalShape[1]):

					delay = calcDelay(x, y, preImg[x][y], globalHeight, globalWidth)

					#print (preImg[x][y])
					prevDelayFrame = int(np.floor(delay))
					nextDelayFrame = int(np.ceil(delay))
					#print delay
					#print (delay, prevDelayFrame, nextDelayFrame)
					#postImg[x][y] = Buffer[curFrame-delay][x][y]
					
					if(curFrame < delay):
						#postImg[x][y] = Buffer[curFrame][x][y]
						postImg[x][y] = np.array([0,0,0])
					else:
						pixelOut = pixelWeightedAvg(Buffer[curFrame - prevDelayFrame][x][y], Buffer[curFrame - nextDelayFrame][x][y], delay-prevDelayFrame)
						postImg[x][y] = np.array(pixelOut)
						#postImg[x][y] = [0,0,0]
					
			scipy.misc.imsave('outframes' + filename[-7:-4]  + '.jpg', postImg)
			curFrame += 1


def writeMovie():
	pathIn= '/Users/elishieber/Desktop/Thanksgiving/outframes'

	img_array = []
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	#print files
	files.sort(key = lambda x: x[5:-4])
	for filename in files:
		if (filename != ".DS_Store"):
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)
	 
	os.chdir("/Users/elishieber/Desktop/Thanksgiving/outmovie")
	out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)   
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

def pixelWeightedAvg(p1, p2, p1weight):
	#print "p1 ----> " + str(p1)
	rOut = p1[0]*(1-p1weight) + p2[0]*p1weight
	gOut = p1[1]*(1-p1weight) + p2[1]*p1weight
	bOut = p1[2]*(1-p1weight) + p2[2]*p1weight
	#print "rOut ----> " + str(rOut)
	pixelOut = [int(rOut), int(gOut), int(bOut)]
	return pixelOut

def calcDelay(x, y, color, width, height):
	#delay = (x+y)/16.
	#delay = np.abs(30.*np.sin((x+y)/32.))
	#delay = np.abs((x - y)/50. + 360.)
	
	R = color[0]
	G = color[1]
	B = color[2]
	
	#delay = np.sqrt(((x - (width/2.))/32.)**2 + ((y - (height/2.))/32.)**2)
	#print(R)
	delay = (R + G + B)/64.
	return delay

def calcMaxDelay(dimX, dimY):
	curMax = 0
	for i in range(0, dimX):
		for j in range(0, dimY):
			curDelay = calcDelay(i, j)
			if (curDelay > curMax):
				curMax = curDelay
	return curMax


'''
print("cleaning up")
cleanUp()

print("initializing")
init()

print("reading movie")
readMovie()

print("applying rolling shutter")
rollingShutter()

print("writing movie")
writeMovie()
'''

writeMovie()


