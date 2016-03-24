# Deel
Deel; A High level deep neural network description language.

***Now Under construction***

![logo](deel.png)


## Goal
Describe deep neural network, training and using in simple syntax.

###Examples

####CNN classifier (done)
```python
CNN = GoogLeNet()

Input("deel.png")
CNN.classify()
Show()

```

####CNN trainer (done / but only for CPU)
```python
nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
			val="data/test.txt")

def workout(x,t):
	nin.classify(x)	
	nin.backprop(t)

BatchTrain(workout)
```

####CNN-LSTM trainer (not yet)
```python
CNN = GoogLeNet()
RNN = LSTM()

InputBatch(train='train.tsv',
			test='test.tsv')
def workout(x,t):
	Input(x)
	CNN.classify() 
	RNN.forward()
	RNN.backprop(t)

BatchTrain(epoch=500)
```

####CNN-DQN with Unity (not yet)
```python
CNN = GoogLeNet()
DQN = DeepQLearning()

def workout():
	#Realtime input image from Unity
	InputStream('unity.png') 
	CNN.classify() 
	DQN.forward()

	#Get score or loss from Unity game
	t = InputVarsFromUnity()
	DQN.reinforcement(t)

StreamTrain(workout)
```

