# Deel
Deel; A High level deep neural network description language.

You can create your own deep neural network application in a second.

This is Deel supported Python 3.x.

![logo](deel.png)


## Goal
Describe deep neural network, training and using in simple syntax.

## Dependency

Chainer 1.7.1 or higher

Python 3.5.2 or heigher

(Optional) OpenCv 2.4.12 or higher

## Install and test

```sh
$ git clone https://github.com/uei/deel.git
$ cd deel
$ python setup.py install
$ cd deel/data
$ ./getCaltech101.sh
$ cd ../misc
$ ./getPretrainedModels.sh
$ cd ..
$ python test.py
```

###Examples

####CNN classifier 
```python
from deel.deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN = GoogLeNet()

CNN.Input("deel.png")
CNN.classify()
ShowLabels()

```

####CNN trainer 
```python
from deel.deel import *
from deel.network import *
from deel.commands import *

nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
			val="data/test.txt")

def workout(x,t):
	nin.classify(x)	
	return nin.backprop(t)

BatchTrain(workout)
```

####CNN classifier with OpenCV camera (you need OpenCV2) 
```python
import cv2 
from deel.deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN = GoogLeNet()

cam = cv2.VideoCapture(0)  

while True:
	ret, img = cam.read()  
	CNN.Input(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	CNN.classify()

	labels = GetLabels()
	if labels[0][1] == 'Band':
		print 'BAND'
		cv2.imwrite('band.png',img)

	cv2.imshow('cam', img)
	if cv2.waitKey(10) > 0:
		break
cam.release()
cv2.destroyAllWindows()

```



####CNN-DQN with Unity (using with https://github.com/wbap/ml-agent-for-unity)
```python
from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *

deel = Deel()

CNN = AlexNet()
QNET = DQN()

def trainer(x):
	CNN.feature(x)
	return QNET.actionAndLearn()

StartAgent(trainer)
```


####ResNet Inferrence
```python
from deel.deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.network.resnet152 import *
from deel.commands import *
import time
deel = Deel()

CNN = ResNet152()
CNN.Input("test.jpg")
CNN.classify()
ShowLabels()

```

####ResNet Finetuning
```python
from deel.deel import *
from deel.network import *
from deel.commands import *
from deel.network.resnet152 import *
#from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel(gpu=-1)

CNN = ResNet152()

InputBatch(train="data/train.txt",
            val="data/test.txt")

def workout(x,t):
   CNN.batch_feature(x,t) 
   return CNN.backprop(t)

def checkout():
   CNN.save('model_google_cpu.hdf5')

BatchTrain(workout,checkout)
```
