import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from main import evaluate_nca, load_checkpoint, data_func, kwargs

DATA_NUM = 5
to_test = np.arange(1,26+1,1)

winner_flat = load_checkpoint()

training_data, target_data = [], []
for _ in range(DATA_NUM):
    x, y = data_func(**kwargs)
    training_data.append(x)
    target_data.append(y)

result_loss = []
result_acc = []
for test_size in tqdm(to_test):
    loss_sum = 0
    acc_sum = 0
    for i in range(DATA_NUM):
        val, acc = evaluate_nca(
            winner_flat, 
            training_data[i], target_data[i], 
            verbose=False, 
            visualize=False,
            N_neo=test_size, M_neo=test_size, 
            return_accuracy=True
        )
        loss_sum += val
        acc_sum += acc
    result_loss.append(loss_sum / DATA_NUM)
    result_acc.append(float(acc_sum) / float(DATA_NUM))

plt.xticks(to_test, to_test)
plt.plot(to_test, result_loss, label="Loss")
plt.plot(to_test, result_acc, label="Accuracy")
plt.xlabel("NCA size (^2)")
plt.legend()
plt.show()