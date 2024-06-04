
# Perceptrons-Multi-Camada

## Demonstração

Link Google Colab: https://colab.research.google.com/drive/1qZRzm47BTReJhXL9b-gwIUAjGyzZWwJV?usp=sharing

### Introdução

Este projeto demonstra um MLP (Perceptron Multicamadas) desenvolvido para solucionar o problema da porta lógica XOR. O MLP é treinado com o algoritmo de retropropagação e utiliza a função de ativação sigmoide. Adicionalmente, uma versão implementada com a biblioteca PyTorch também está incluída.

####  Classe MLP

```python
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        # Inicialização dos tamanhos das camadas e taxa de aprendizado
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos e bias
        self.weights_input_hidden = np.random.randn(self.input_dim, self.hidden_dim)
        self.bias_hidden = np.zeros((1, self.hidden_dim))
        self.weights_hidden_output = np.random.randn(self.hidden_dim, self.output_dim)
        self.bias_output = np.zeros((1, self.output_dim))

    def sigmoid(self, x):
        # Função de ativação sigmoid
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivada da função sigmoid
        return x * (1 - x)

    def mean_squared_error(self, y_true, y_pred):
        # Função de custo: erro quadrático médio
        return np.mean((y_true - y_pred) ** 2)
    
    def forward(self, X):
        # Passo forward (inferência)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Passo backward (retropropagação)
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Atualização dos pesos e bias
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        # Treinamento do MLP
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if (epoch + 1) % 1000 == 0:
                loss = self.mean_squared_error(y, output)
                print(f'Época {epoch + 1}, Erro Quadrático Médio: {loss}')

# Dados de entrada e saída para a porta XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Criação e treinamento do MLP
mlp = MLP(input_dim=2, hidden_dim=2, output_dim=1, learning_rate=0.1)
mlp.train(X, y)

# Testando a rede
for x in X:
    print(f'Entrada: {x}, Saída Prevista: {mlp.forward(x)}')
```

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definição do modelo MLP em PyTorch
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Camada oculta com 2 neurônios
        self.output = nn.Linear(2, 1)  # Camada de saída com 1 neurônio
    
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # Ativação sigmoid na camada oculta
        x = torch.sigmoid(self.output(x))  # Ativação sigmoid na camada de saída
        return x
```
# Dados de entrada e saída para a porta XOR
```
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
```
# Inicialização do modelo, função de perda e otimizador
```
model = XOR_MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```
# Treinamento do modelo
```
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f'Época {epoch + 1}, Erro: {loss.item()}')
```

# Testando o modelo
```
with torch.no_grad():
    for x in X:
        output = model(x)
        print(f'Entrada: {x.numpy()}, Saída Prevista: {output.numpy()}')
```

### Instalação
Para executar este projeto no Google Colab, não é necessário instalar nada localmente. Basta acessar o link abaixo para o Google Colab e executar o projeto.




