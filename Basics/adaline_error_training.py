import matplotlib.pyplot as plt

X_training = [
    [2.215, 2.063],
    [0.224, 1.586],
    [0.294, 0.651],
    [2.327, 2.932],
    [2.497, 2.322],
    [0.169, 1.943],
    [1.274, 2.428],
    [1.526, 0.596],
    [2.009, 2.161],
    [1.759, 0.342],
    [1.367, 0.938],
    [2.173, 2.719],
    [0.856, 1.904],
    [2.211, 1.868],
    [1.587, 1.642],
    [0.350, 0.841],
    [1.441, 0.091],
    [0.185, 1.327],
    [2.764, 1.149],
    [1.947, 1.598]
]

y_training = [
    -1, 1, 1, -1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, -1, 1, 1, 1, -1, -1
]

def createMatrix(defaultValue, rowSize, columnSize):
    matrix = []
    for i in range(rowSize):
        row = []
        for j in range(columnSize):
            row.append(defaultValue)
        matrix.append(row)
    return matrix

def calculateOutput(inputVector, weightVector, bias):
    total = 0
    for i in range(len(weightVector)):
        total += weightVector[i] * inputVector[i]
    total += bias
    return total

def updateWeights(weightsVector, inputVector, error, learningRate):
    for i in range(len(weightsVector)):
        weightsVector[i] = weightsVector[i] + inputVector[i] * error * learningRate
    return weightsVector

def updateBias(bias, error, learningRate):
    return bias + error * learningRate

precisionRate = 0.001
bias = 0.0
weightsVector = [0.0] * 2 

learningRate = 0.01
previousMse = 0
mseHistory = []

while True:
    accumlatedError = 0
    for index, coordinates in enumerate(X_training):
        neuronOutput = calculateOutput(coordinates, weightsVector, bias)
        expectedOutput = y_training[index]
        error = expectedOutput - neuronOutput
        updateWeights(weightsVector, coordinates, error, learningRate)
        bias = updateBias(bias, error, learningRate)
        accumlatedError += error**2
    
    mse = accumlatedError / len(X_training)
    mseHistory.append(mse)
    if(abs(mse - previousMse) < precisionRate):
        break
    previousMse = mse

# print(weightsVector)
# print(bias) 

plt.figure()
plt.plot(range(0,len(mseHistory)), mseHistory, '-o')
plt.show()

plt.figure()
for i in range(len(X_training)):
    if(y_training[i] == 1):
        plt.scatter(X_training[i][0], X_training[i][1], color='blue')
    else:
        plt.scatter(X_training[i][0], X_training[i][1], color='red')
x_line = [0.0, 3.0]
y_line = [(-weightsVector[0] * x - bias) / weightsVector[1] for x in x_line]
plt.plot(x_line, y_line, color='green')
plt.show()
