import math
import random

X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
T = [-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000]

# Hiperparâmetros da Rede
hidden_neurons = 5
learning_rate = 0.1
epochs = 10000
random.seed(42) # Para que você possa reproduzir o mesmo resultado
# Inicialização de Pesos e Bias (aleatórios entre -0.5 e 0.5)
W1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_neurons)] # Pesos Entrada -> Oculta
B1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_neurons)] # Bias Camada Oculta
W2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_neurons)] # Pesos Oculta -> Saída
B2 = random.uniform(-0.5, 0.5)                                  # Bias Saída

# Funções de ativação e derivada
def tanh(x):
    return math.tanh(x)

def tanh_deriv(x):
    return 1.0 - math.tanh(x)**2

# Loop de Treinamento (Stochastic Gradient Descent)
for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(X)):
        x = X[i]
        target = T[i]

        # --- FORWARD PASS (Propagação do sinal) ---
        
        # Camada Oculta
        hidden_outputs = []
        for j in range(hidden_neurons):
            # z = w * x + b
            net_h = W1[j] * x + B1[j]
            hidden_outputs.append(tanh(net_h))

        # Camada de Saída (Ativação linear para regressão)
        y = B2
        for j in range(hidden_neurons):
            y += W2[j] * hidden_outputs[j]

        # Cálculo do Erro Quadrático
        error = y - target
        total_error += error**2

        # --- BACKPROPAGATION (Ajuste dos pesos) ---
        
        # O gradiente da saída para ativação linear é o próprio erro (y - target)
        delta_out = error

        for j in range(hidden_neurons):
            # Delta da camada oculta: (delta_out * peso_saida) * derivada_da_ativacao
            net_h = W1[j] * x + B1[j]
            delta_h = delta_out * W2[j] * tanh_deriv(net_h)

            # Atualização dos pesos da camada de saída
            W2[j] -= learning_rate * delta_out * hidden_outputs[j]

            # Atualização dos pesos e bias da camada oculta
            W1[j] -= learning_rate * delta_h * x
            B1[j] -= learning_rate * delta_h

        # Atualização do bias da saída
        B2 -= learning_rate * delta_out

    # Imprimir o progresso
    if epoch % 1000 == 0:
        mse = total_error / len(X)
        print(f"Época {epoch:05d} | MSE: {mse:.6f}")

#Testando os Resultados Finais
print("\n--- Resultados Finais ---")
for i in range(len(X)):
    x = X[i]
    # Replicando o Forward Pass com os pesos treinados
    hidden_outputs = [tanh(W1[j] * x + B1[j]) for j in range(hidden_neurons)]
    y = B2 + sum([W2[j] * hidden_outputs[j] for j in range(hidden_neurons)])
    
    print(f"x: {x:.1f} | Alvo Real: {T[i]:>7.4f} | Previsão MLP: {y:>7.4f}")