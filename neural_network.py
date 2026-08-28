"""
Below is a Perceptron(Single layer neuron) that solves "OR" problem

Condition:
    y = 0 if I < 0.5
    y = 1 otherwise

Training_Pattern    Attributes          Desired_Output
                    Input_1 Input_2
    1                   1       1           1
    2                   0       1           1
    3                   1       0           1
    4                   0       0           0

    

Thought-process
1. Build a 


Please assume a
random value of weights (ideally between 0 and 1) in the network and a learning rate ()= 0.2

Now, assume the following definition of output
𝑦 =
0 𝑖𝑓 𝐼 < 0.5
1 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
Where, 𝐼 is the overall input at a node in the network.
Please find out whether classes represented in the desired output are linearly separable
and can be modeled by a neural network with one node in a single layer. Please show
your working.
Now, propose a suitable neural network for the dataset above. Please assume a
random value of weights (ideally between 0 and 1) in the network and a learning rate ()
= 0.2 in the neural network. Using case-based updating, please show the change in
weights in each epoch as well as the final set of weights for the neural network.
Finally, please use the final set of weights and the output function to derive predictions
from the neural network for the training dataset shown above. Does the neural network
classify all the training patterns correctly?
Now, derive predictions for the dataset shown below. How many of the patterns below
does the neural network correctly classify?
Training
Pattern
Attributes Desired Output
Input 1 Input 2
1 1 1 0
2 0 1 1
3 1 0 1
4 0 0 0
"""
import random

class Perceptron:
    def __init__(self, learning_rate=0.2, seed=7):
        self.learning_rate = learning_rate
        random_generator = random.Random(seed)
        self.weights = [random_generator.uniform(0, 1) for _ in range(2)]
        self.bias = 0.2

        print("Initialised perceptron")
        print(f"  weights = {self.weights}")
        print(f"  bias    = {self.bias:.4f}")
        print(f"  learning rate = {self.learning_rate}")

    def net_input(self, inputs):
        return sum(weight * value for weight, value in zip(self.weights, inputs)) + self.bias

    def predict(self, inputs):
        # 1 if x > 0.5
        return int(self.net_input(inputs) >= 0.5)

    def train(self, training_data, max_epochs=100):
        """Train with case-based (pattern-by-pattern) perceptron updates."""
        for epoch in range(1, max_epochs + 1):
            weights_before = self.weights[:]
            bias_before = self.bias
            mistakes = 0

            print(f"\nEpoch {epoch}")
            for inputs, desired_output in training_data:
                output = self.predict(inputs)
                error = desired_output - output
                old_weights = self.weights[:]
                old_bias = self.bias

                if error != 0:
                    self.weights = [
                        weight + self.learning_rate * error * value
                        for weight, value in zip(self.weights, inputs)
                    ]
                    self.bias += self.learning_rate * error
                    mistakes += 1

                weight_delta = [new - old for new, old in zip(self.weights, old_weights)]
                bias_delta = self.bias - old_bias
                print(
                    f"  {inputs} -> desired={desired_output}, predicted={output}, "
                    f"delta_w={weight_delta}, delta_b={bias_delta:.4f}, "
                    f"weights={self.weights}, bias={self.bias:.4f}"
                )

            epoch_weight_delta = [
                new - old for new, old in zip(self.weights, weights_before)
            ]
            epoch_bias_delta = self.bias - bias_before
            print(
                f"  Epoch change: delta_w={epoch_weight_delta}, "
                f"delta_b={epoch_bias_delta:.4f}, mistakes={mistakes}"
            )

            if mistakes == 0:
                print("  No mistakes: training has converged.")
                return epoch

        print(f"Training stopped after {max_epochs} epochs without convergence.")
        return max_epochs

    def show_predictions(self, dataset, title):
        print(f"\n{title}")
        correct = 0
        for inputs, desired_output in dataset:
            activation = self.net_input(inputs)
            prediction = self.predict(inputs)
            is_correct = prediction == desired_output
            correct += int(is_correct)
            print(
                f"  {inputs}: I={activation:.4f}, predicted={prediction}, "
                f"desired={desired_output}, correct={is_correct}"
            )
        print(f"Correctly classified: {correct}/{len(dataset)}")
        return correct


OR_DATA = [
    ([1, 1], 1),
    ([0, 1], 1),
    ([1, 0], 1),
    ([0, 0], 0),
]

SECOND_DATA = [
    ([1, 1], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([0, 0], 0),
]


if __name__ == "__main__":
    print("LINEAR SEPARABILITY")
    print("The OR classes are linearly separable by one threshold node.")
    print(
        "The second dataset is XOR: no single straight decision boundary can separate its classes."
    )

    perceptron = Perceptron(learning_rate=0.2, seed=7)
    perceptron.train(OR_DATA)

    print("\nFinal parameters")
    print(f"  weights = {perceptron.weights}")
    print(f"  bias    = {perceptron.bias:.4f}")
    perceptron.show_predictions(OR_DATA, "Predictions for OR training data")
    perceptron.show_predictions(SECOND_DATA, "Predictions for the second dataset(XOR)")