import matplotlib.pyplot as plt
import math

x_data = [
    -3.4557427, -3.4822109, -3.1973477, -2.9263939, -1.9737541, -1.9189151,
    -1.9781052, -1.0670384, -1.4996849, -1.1159294, -0.781221, -0.5649162,
    -0.0050037, 0.1969329, 0.3321719, 0.8792238, 1.0729627, 1.2940016,
    1.4121696, 2.0611459, 2.1194811, 2.9184708, 2.3294477, 3.0532795,
    3.1210985, 3.5576675, 3.7092212, 3.8422916, 4.7783129, 4.4976173,
    4.683564, 5.5428325, 5.2959788, 6.1256565, 6.5558376, 6.4062025,
    6.6951968, 7.4498412, 6.9709788, 7.3426909, 7.9904375, 8.3039034,
    8.685398, 9.1763368, 9.0756499, 9.2065044, 9.530235, 10.350861,
    10.663104, 10.343534
]

y_data = [
    -1.6239881, -1.0618243, -1.3302606, -0.8884057, -0.9198156, -0.4542355,
    -0.2378872, -0.451436, -0.1417166, 0.464144, 0.8016596, 0.2934546,
    1.1213746, 0.8547785, 1.2483002, 1.8912561, 1.8531781, 1.9523057,
    1.8967559, 2.7176396, 2.2022322, 2.5712507, 3.3505474, 3.5432877,
    3.4236653, 4.0951618, 3.9549787, 4.5025232, 3.7643277, 4.666918,
    4.5328549, 4.9349832, 5.3850333, 4.7989586, 5.7666838, 6.0690915,
    5.9136599, 6.1221843, 6.5711131, 6.0160765, 6.722859, 6.492281,
    7.2993508, 7.0432869, 7.5261253, 8.1722142, 8.1273208, 7.595554,
    8.3859757, 8.3096467
]

precisionRate = 0.0001
bias = 0.0
w = 0.0 

learningRate = 0.001 
previousMse = float('inf')
mseHistory = []

epochs = 0
max_epochs = 5000 

while epochs < max_epochs:
    accumulatedError = 0
    for i in range(len(x_data)):
        neuronOutput = w * x_data[i] + bias 
        expectedOutput = y_data[i]
        error = expectedOutput - neuronOutput
        
        w += learningRate * error * x_data[i]
        bias += learningRate * error
        
        accumulatedError += error**2
    
    mse = accumulatedError / len(x_data)
    mseHistory.append(mse)
    
    if abs(previousMse - mse) < precisionRate:
        break
        
    previousMse = mse
    epochs += 1

n = len(x_data)
mean_x = sum(x_data) / n
mean_y = sum(y_data) / n

numerator = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
denominator = sum((x_data[i] - mean_x)**2 for i in range(n))

a_analytical = numerator / denominator
b_analytical = mean_y - a_analytical * mean_x

num_pearson = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
den_pearson = math.sqrt(sum((x_data[i] - mean_x)**2 for i in range(n)) * sum((y_data[i] - mean_y)**2 for i in range(n)))
r_pearson = num_pearson / den_pearson

sst = sum((y_data[i] - mean_y)**2 for i in range(n)) 
ssr = sum((y_data[i] - (w * x_data[i] + bias))**2 for i in range(n)) 
r_squared = 1 - (ssr / sst)

print("--- RESULTADOS ---")
print(f"Épocas de treino: {epochs}")
print(f"Adaline Encontrou -> Peso (w): {w:.5f} | Viés (b): {bias:.5f}")
print(f"Eq Analítica    -> a: {a_analytical:.5f} | b: {b_analytical:.5f}")
print(f"Coeficiente de Pearson (r): {r_pearson:.5f}")
print(f"Coeficiente de Determinação (R²): {r_squared:.5f}")

# Gráfico de Erro (MSE)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Evolução do Erro (MSE)")
plt.plot(range(len(mseHistory)), mseHistory, color='purple')
plt.xlabel("Épocas")
plt.ylabel("MSE")

# Gráfico da Regressão Linear
plt.subplot(1, 2, 2)
plt.title("Regressão Linear: Adaline vs Analítica")
plt.scatter(x_data, y_data, color='blue', label='Dados')

# Pontos extremos para traçar as retas
x_min, x_max = min(x_data), max(x_data)

# Reta do Adaline
y_min_ada = w * x_min + bias
y_max_ada = w * x_max + bias
plt.plot([x_min, x_max], [y_min_ada, y_max_ada], color='red', linewidth=3, label=f'Adaline: {w:.2f}x + {bias:.2f}')

# Reta Analítica (ax + b)
y_min_analyt = a_analytical * x_min + b_analytical
y_max_analyt = a_analytical * x_max + b_analytical
plt.plot([x_min, x_max], [y_min_analyt, y_max_analyt], color='yellow', linestyle='dashed', linewidth=2, label=f'Eq: {a_analytical:.2f}x + {b_analytical:.2f}')

plt.legend()
plt.tight_layout()
plt.show()