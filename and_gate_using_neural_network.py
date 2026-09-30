"""
AND GATE
    -   1 only if all 1

Steps in perceptron Algorithm
    1. Initialise weights values and biases
    2. forward propogate the values and biases
    3. check the error
    4. Backpropogate and adjust weight and biases
    5. Repeat for all training examples    

"""
import random

random.seed(42)

train = [(0,0,0),
         (0,1,0),
         (1,0,0),
         (1,1,1)
         ]
print(train)


def op_calc(w1: float,w2: float, b: float, x1: int, x2: int) -> float:
    result = (w1*x1) + (w2*x2) + b
    return result

def predict(t: float) -> int:
    return 1 if t >= 0 else 0

def update_weights(w: float, e: float, x: float) -> float:
    n = 0.1                 # n-> learning rate (hyperparameter)
    delta_w = n * e * x
    w = w + delta_w
    return w

def update_bias(b: float, e: float) -> float:
    n = 0.1                 # n-> learning rate (hyperparameter)
    delta_b = n * e
    b = b + delta_b
    return b

def trainz(train: list):
    # initalise weights
    w1 = 0.0
    w2 = 0.0
    b = 0.0
    epochs = 10

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        cnt = 0
        for x1,x2,o in train:
            t = op_calc(w1,w2,b,x1,x2)
            p = predict(t)
            # rightnow we are checking error each time and updating weights and biases
            print(f"x1: {x1}, x2: {x2}, actual: {o}, pred: {p}")
            if p != o:
                cnt += 1
                e = o-p
                w1,w2 = update_weights(w1,e,x1),update_weights(w2,e,x2)
                b = update_bias(b,e)

        if cnt == 0:    # all errors solved
            for x1,x2,o in train:
                t = op_calc(w1,w2,b,x1,x2)
                if predict(t) != o:
                    print("Still Incorrect Output")
                    break
            print("Training done")
            break

trainz(train)


    