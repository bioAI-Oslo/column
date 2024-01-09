import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

print("Hello and welcome to test script. I just want to generate memory and see how it affects the VM")

ITERATIONS = 100000
SAMPLES = 5000

(train_X, train_y), (test_X, test_y) = mnist.load_data()

data = np.empty(ITERATIONS)
for iter in tqdm(range(ITERATIONS)):

    random_sample = np.random.randint(0,len(train_X),size=SAMPLES)
    x_data = train_X[random_sample]
    y_data = train_y[random_sample]
    data[iter] = np.mean(x_data)

print(np.mean(data))

