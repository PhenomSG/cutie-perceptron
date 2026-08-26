"""
plan
1. what is perceptron? -> 
		A small learning algorithm that separates two groups
   			of data by using weights and a bias.
2. how i plan to implement it -> 

		i have no plan ... bratha
		idk start with some weight

		Start with zero weights, make predictions,
   		compare them with the correct answers, and adjust the weights when a
   		prediction is wrong.
"""

# gfg implementation + blog

class Perceptron:
	def __init__(self, learning_rate=0.1):
		self.learning_rate = learning_rate
		self.weights = []
		self.bias = 0

	def predict(self, inputs):
		total = self.bias

		for input_value, weight in zip(inputs, self.weights):
			total += input_value * weight

		if total >= 0:
			return 1
		return 0

	def train(self, training_inputs, answers, rounds=10):
		self.weights = [0] * len(training_inputs[0])

		for _ in range(rounds):


			# check this part
			for inputs, answer in zip(training_inputs, answers):
				prediction = self.predict(inputs)
				error = answer - prediction

				for index in range(len(self.weights)):
					self.weights[index] += (
						self.learning_rate * error * inputs[index]
					)

				self.bias += self.learning_rate * error


training_inputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
]

answers = [0, 1, 1, 1]

perceptron = Perceptron()
perceptron.train(training_inputs, answers)

for inputs in training_inputs:
	print(inputs, "->", perceptron.predict(inputs))
