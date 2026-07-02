import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1.0
    return one_hot

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def compute_loss(Y, Y_hat):
    m = Y.shape[0]
    epsilon = 1e-10
    loss = -np.sum(Y * np.log(Y_hat + epsilon)) / m
    return loss

def get_predictions(A2):
    return np.argmax(A2, axis=1)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.shape[0]

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        
        return self.A2
        
    def backward_propagation(self, X, Y):
        m = X.shape[0]
        
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
        
    def update_params(self, dW1, db1, dW2, db2, learning_rate, momentum=0.9):
        self.v_W1 = momentum * self.v_W1 - learning_rate * dW1
        self.v_b1 = momentum * self.v_b1 - learning_rate * db1
        self.v_W2 = momentum * self.v_W2 - learning_rate * dW2
        self.v_b2 = momentum * self.v_b2 - learning_rate * db2

        self.W1 += self.v_W1
        self.b1 += self.v_b1
        self.W2 += self.v_W2
        self.b2 += self.v_b2

    def fit(self, X_train, Y_train_oh, Y_train_lbl, X_val, Y_val_oh, Y_val_lbl, epochs, learning_rate, batch_size, patience=10, momentum=0.9):
        m_train = X_train.shape[0]
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        
        for epoch in range(epochs):
            epoch_perm = np.random.permutation(m_train)
            X_tr = X_train[epoch_perm]
            Y_tr_oh = Y_train_oh[epoch_perm]
            Y_tr_lbl = Y_train_lbl[epoch_perm]
            
            epoch_loss = 0.0
            epoch_correct = 0
            
            for i in range(0, m_train, batch_size):
                X_batch = X_tr[i:i+batch_size]
                Y_batch_oh = Y_tr_oh[i:i+batch_size]
                Y_batch_lbl = Y_tr_lbl[i:i+batch_size]
                
                A2_batch = self.forward_propagation(X_batch)
                
                batch_loss = compute_loss(Y_batch_oh, A2_batch)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                preds = get_predictions(A2_batch)
                epoch_correct += np.sum(preds == Y_batch_lbl)
                
                dW1, db1, dW2, db2 = self.backward_propagation(X_batch, Y_batch_oh)
                
                self.update_params(dW1, db1, dW2, db2, learning_rate, momentum)
                
            train_loss = epoch_loss / m_train
            train_acc = epoch_correct / m_train
            
            A2_val = self.forward_propagation(X_val)
            val_loss = compute_loss(Y_val_oh, A2_val)
            val_preds = get_predictions(A2_val)
            val_acc = get_accuracy(val_preds, Y_val_lbl)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc * 100)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc * 100)
            
            # Print epoch information (disabled to avoid too much output, but can be useful for debugging)
            # print(f"Época {epoch+1:03d}/{epochs} | Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping acionado na época {epoch+1}. Restaurando pesos da melhor época.")
                    self.W1, self.b1, self.W2, self.b2 = best_weights
                    break
            
        return history

    def predict(self, X):
        A2 = self.forward_propagation(X)
        return get_predictions(A2)

def main():
    iris = fetch_ucirepo(id=53) 
    
    # Extraindo as features e labels do pandas dataframe para numpy arrays
    X = iris.data.features.to_numpy()
    y_raw = iris.data.targets.to_numpy().flatten()
    
    # Mapear strings para inteiros (ex: 'Iris-setosa' -> 0)
    classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_raw])
    
    m = X.shape[0]
    
    # Embaralhar o dataset
    np.random.seed(42) # semente fixa para reprodutibilidade
    permutation = np.random.permutation(m)
    X = X[permutation]
    y = y[permutation]
    
    train_end = int(m * 0.7)
    val_end = int(m * 0.85)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    # print(f"Total de amostras: {m}")
    # print(f"Treino: {X_train.shape[0]} amostras")
    # print(f"Validação: {X_val.shape[0]} amostras")
    # print(f"Teste: {X_test.shape[0]} amostras\n")
    
    # Normalização das features (Standardization: média 0 e variância 1) usando dados de treino
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # One-hot encoding
    num_classes = len(classes)
    y_train_oh = one_hot_encode(y_train, num_classes)
    y_val_oh = one_hot_encode(y_val, num_classes)
    y_test_oh = one_hot_encode(y_test, num_classes)
    
    INPUT_SIZE = X_train.shape[1]
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = num_classes
    LEARNING_RATE = 0.05
    EPOCHS = 500
    BATCH_SIZE = 16
    PATIENCE = 30
    
    print(f"Camada Oculta: {HIDDEN_SIZE} neurônios")
    print(f"Taxa de Aprendizado: {LEARNING_RATE}")
    print(f"Tamanho do Lote (Batch): {BATCH_SIZE}")
    print(f"Patience (Early Stopping): {PATIENCE}\n")
    
    mlp = MultilayerPerceptron(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    print("Iniciando o Treinamento...")
    history = mlp.fit(X_train, y_train_oh, y_train, 
                      X_val, y_val_oh, y_val, 
                      epochs=EPOCHS, learning_rate=LEARNING_RATE, 
                      batch_size=BATCH_SIZE, patience=PATIENCE)
    
    print("\nTestando a Rede Neural")
    test_predictions = mlp.predict(X_test)
    test_accuracy = get_accuracy(test_predictions, y_test)
    
    print(f"Acurácia Final no Conjunto de Teste: {test_accuracy * 100:.2f}%\n")
    
    plt.figure(figsize=(12, 5))
    actual_epochs = len(history['train_loss'])
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), history['train_loss'], color='red', label='Treino', marker='o', markersize=3)
    plt.plot(range(1, actual_epochs + 1), history['val_loss'], color='orange', label='Validação', marker='s', markersize=3)
    plt.title('Loss: Treino vs Validação')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), history['train_acc'], color='blue', label='Treino', marker='o', markersize=3)
    plt.plot(range(1, actual_epochs + 1), history['val_acc'], color='cyan', label='Validação', marker='s', markersize=3)
    plt.title('Acurácia: Treino vs Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()