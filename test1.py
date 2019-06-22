import pyttsx3
import time

engine = pyttsx3.init()

#运行并且等待
#

while True:
	localtime = time.strftime("%S", time.localtime())
	print(localtime)
	engine.say(localtime)
	print(localtime)
	engine.runAndWait()