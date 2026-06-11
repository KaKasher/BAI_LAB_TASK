import json

file_path = '/Users/kaka/uni/BAI_LAB_TASK/lab_3/Lab3_MLP_Backpropagation_MAIN_FILE.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

task1_code = """import numpy as np

def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

# x - sygnal wejsciowy [1, x1, x2, ..., xN]
# w1 - wagi warstwy ukrytej, macierz (K x N+1)
# w2 - wagi warstwy wyjsciowej, wektor (1 x K+1)
# beta - parametr funkcji aktywacji
def mlp(x, w1, w2, beta):
    v_hidden = tanh(np.dot(w1, x), beta)
    v = np.concatenate(([1.0], v_hidden))
    y = sigmoid(np.dot(w2, v), beta)
    return y, v, v_hidden

# Test: losowe wagi, wejscie [1, 0, 1]
# np.random.seed(0)
# w1_test = np.random.randn(2, 3) * 0.5
# w2_test = np.random.randn(1, 3) * 0.5
# result = mlp(np.array([1, 0, 1]), w1_test, w2_test, beta=1.0)
# print(f"Wyjscie sieci: {result[0]}")
"""

task2_code = """import numpy as np
import matplotlib.pyplot as plt

# Dane XOR (z biasem, sygnaly niezerowe!)
xx = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
d = np.array([0, 1, 1, 0])

def sigmoid_diff(y, beta):
    return beta * y * (1 - y)

def tanh_diff(y, beta):
    return beta * (1 - y * y)

def train_sample(xx, d, eta, beta):
    \"\"\"Wariant 1: aktualizacja wag po KAZDEJ probce.\"\"\"
    np.random.seed(42)
    w1 = np.random.randn(2, 3) * 0.5
    w2 = np.random.randn(1, 3) * 0.5
    
    errors = []
    
    for epoch in range(100000):
        total_error = 0.0
        misclassified = 0
        
        for i in range(len(xx)):
            x = xx[i]
            target = d[i]
            
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y_val = y[0]
            
            error = target - y_val
            total_error += error ** 2
            
            pred = 1 if y_val >= 0.5 else 0
            if pred != target:
                misclassified += 1
                
            delta_out = error * sigmoid_diff(y_val, beta)
            delta_hidden = delta_out * w2[0, 1:] * tanh_diff(v_hidden, beta)
            
            w2[0, :] += eta * delta_out * v
            w1 += eta * np.outer(delta_hidden, x)
            
        errors.append(total_error / len(xx)) # MSE
        
        if misclassified == 0:
            break
            
    return errors

def train_epoch(xx, d, eta, beta):
    \"\"\"Wariant 2: aktualizacja wag po EPOCE.\"\"\"
    np.random.seed(42)
    w1 = np.random.randn(2, 3) * 0.5
    w2 = np.random.randn(1, 3) * 0.5
    
    errors = []
    
    for epoch in range(100000):
        total_error = 0.0
        misclassified = 0
        
        grad_w1 = np.zeros_like(w1)
        grad_w2 = np.zeros_like(w2)
        
        for i in range(len(xx)):
            x = xx[i]
            target = d[i]
            
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y_val = y[0]
            
            error = target - y_val
            total_error += error ** 2
            
            pred = 1 if y_val >= 0.5 else 0
            if pred != target:
                misclassified += 1
                
            delta_out = error * sigmoid_diff(y_val, beta)
            delta_hidden = delta_out * w2[0, 1:] * tanh_diff(v_hidden, beta)
            
            grad_w2[0, :] += delta_out * v
            grad_w1 += np.outer(delta_hidden, x)
            
        w2 += eta * grad_w2
        w1 += eta * grad_w1
        
        errors.append(total_error / len(xx)) # MSE
        
        if misclassified == 0:
            break
            
    return errors

# Trening i wykres
errors_sample = train_sample(xx, d, eta=0.5, beta=1.0)
errors_epoch = train_epoch(xx, d, eta=0.5, beta=1.0)

plt.figure(figsize=(10, 5))
plt.plot(errors_sample, label='Po kazdej probce', alpha=0.8)
plt.plot(errors_epoch, label='Po epoce', alpha=0.8)
plt.xlabel('Epoka')
plt.ylabel('Blad MSE')
plt.title('Porownanie dwoch wariantow aktualizacji wag')
plt.legend()
plt.yscale('log')
plt.grid(alpha=0.3)
plt.show()
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'def mlp(x, w1, w2, beta):' in source_str:
            cell['source'] = [line + '\n' for line in task1_code.split('\n')][:-1]
        elif 'def train_sample(xx, d, eta, beta):' in source_str:
            cell['source'] = [line + '\n' for line in task2_code.split('\n')][:-1]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfuly!")
