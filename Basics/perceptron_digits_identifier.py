def createMatrix(defaultValue, rowSize, columnSize):
    matrix = []
    for i in range(rowSize):
        row = []
        for j in range(columnSize):
            row.append(defaultValue)
        matrix.append(row)
    return matrix

def calculateOutput(digitVector, weightVector, bias):
    total = 0
    for i in range(len(weightVector)):
        total += weightVector[i] * digitVector[i]
    total += bias
    return 1 if total > 0 else -1

def calculateTestingOutput(digitVector, weightVector, bias):
    total = 0
    for i in range(len(weightVector)):
        total += weightVector[i] * digitVector[i]
    total += bias
    return total

def updateWeights(weightsVector, digitVector, error, learningRate):
    for i in range(len(weightsVector)):
        weightsVector[i] = weightsVector[i] + digitVector[i] * error * learningRate
    return weightsVector

def updateBias(bias, error, learningRate):
    return bias + error * learningRate

def testNeuralNet(X_testing, finalWeightsMatrix, biasVector):
    resultsMatrix = []
    for digit in X_testing:
        outputsVector = []
        for neuronIndex, neuron in enumerate(finalWeightsMatrix):
            outputsVector.append(calculateTestingOutput(digit,neuron,biasVector[neuronIndex]))
        resultsMatrix.append(outputsVector)
    
    for i in range(len(resultsMatrix)):
        print(f'Digit: {i} Neuron: {resultsMatrix[i].index(max(resultsMatrix[i]))}')

# 0
matrix_0 = [-1,  1,  1, -1,
             1, -1, -1,  1,
             1, -1, -1,  1,
             1, -1, -1,  1,
            -1,  1,  1, -1]

# 1
matrix_1 = [-1,  1, -1, -1,
             1,  1, -1, -1,
            -1,  1, -1, -1,
            -1,  1, -1, -1,
             1,  1,  1, -1]

# 2
matrix_2 = [-1,  1,  1, -1,
             1, -1, -1,  1,
            -1, -1,  1, -1,
            -1,  1, -1, -1,
             1,  1,  1,  1]

# 3
matrix_3 = [ 1,  1,  1, -1,
            -1, -1, -1,  1,
            -1,  1,  1, -1,
            -1, -1, -1,  1,
             1,  1,  1, -1]

# 4
matrix_4 = [ 1, -1, -1,  1,
             1, -1, -1,  1,
             1,  1,  1,  1,
            -1, -1, -1,  1,
            -1, -1, -1,  1]

# 5
matrix_5 = [ 1,  1,  1,  1,
             1, -1, -1, -1,
             1,  1,  1, -1,
            -1, -1, -1,  1,
             1,  1,  1, -1]

# 6
matrix_6 = [-1,  1,  1,  1,
             1, -1, -1, -1,
             1,  1,  1, -1,
             1, -1, -1,  1,
            -1,  1,  1, -1]

# 7
matrix_7 = [ 1,  1,  1,  1,
            -1, -1, -1,  1,
            -1, -1,  1, -1,
            -1,  1, -1, -1,
             1, -1, -1, -1]

# 8
matrix_8 = [-1,  1,  1, -1,
             1, -1, -1,  1,
            -1,  1,  1, -1,
             1, -1, -1,  1,
            -1,  1,  1, -1]

# 9
matrix_9 = [-1,  1,  1, -1,
             1, -1, -1,  1,
            -1,  1,  1,  1,
            -1, -1, -1,  1,
            -1,  1,  1, -1]

X_training = [
    matrix_0, matrix_1, matrix_2, matrix_3, matrix_4,
    matrix_5, matrix_6, matrix_7, matrix_8, matrix_9
]

biasVector = [0] * 10
weightsMatrix = createMatrix(0, 10, 20)
learningRate = 0.1

outputsMatrix = []

for i in range(10):
    row = []
    for j in range(10):
        if(i == j):
            row.append(1)
        else:
            row.append(-1)
    outputsMatrix.append(row)


while True:
    totalErrors = 0
    for imageIndex, digitVector in enumerate(X_training):
        for neuronIndex, neuron in enumerate(weightsMatrix):
            neuronBias = biasVector[neuronIndex]
            
            calculatedOutput = calculateOutput(digitVector, neuron, neuronBias)
            
            expectedOutput = outputsMatrix[imageIndex][neuronIndex]
            
            if(calculatedOutput != expectedOutput):
                totalErrors += 1
                
                error = expectedOutput - calculatedOutput
                
                neuron = updateWeights(neuron, digitVector, error, learningRate)
                
                biasVector[neuronIndex] = updateBias(neuronBias, error, learningRate)
    if(totalErrors == 0): break

##print(weightsMatrix)
print()

noisy_0 = [-1,  1,  1, -1,
            1, -1, -1,  1,
            1,  1, -1,  1, 
            1, -1, -1,  1,
           -1,  1,  1, -1]

noisy_1 = [-1,  1, -1, -1,
            1,  1, -1, -1,
           -1,  1, -1, -1,
           -1,  1, -1, -1,
            1, -1,  1, -1]

noisy_2 = [-1,  1,  1, -1,
            1, -1, -1,  1,
           -1, -1, -1, -1, 
           -1,  1, -1, -1,
            1,  1,  1,  1]

noisy_3 = [ 1,  1,  1, -1,
           -1, -1, -1,  1,
           -1, -1,  1, -1, 
           -1, -1, -1,  1,
            1,  1,  1, -1]

noisy_4 = [ 1, -1,  1,  1, 
            1, -1, -1,  1,
            1,  1,  1,  1,
           -1, -1, -1,  1,
           -1, -1, -1,  1]

noisy_5 = [ 1,  1,  1,  1,
            1, -1, -1, -1,
            1,  1,  1, -1,
            1, -1, -1,  1, 
            1,  1,  1, -1]

noisy_6 = [-1,  1,  1,  1,
            1, -1, -1, -1,
            1,  1,  1, -1,
           -1, -1, -1,  1, 
           -1,  1,  1, -1]

noisy_7 = [ 1, -1,  1,  1, 
           -1, -1, -1,  1,
           -1, -1,  1, -1,
           -1,  1, -1, -1,
            1, -1, -1, -1]

noisy_8 = [-1,  1,  1, -1,
            1, -1, -1,  1,
           -1,  1, -1, -1, 
            1, -1, -1,  1,
           -1,  1,  1, -1]

noisy_9 = [-1,  1,  1, -1,
            1, -1, -1,  1,
           -1,  1,  1,  1,
           -1,  1, -1,  1, 
           -1,  1,  1, -1]

X_testing = [
    noisy_0, noisy_1, noisy_2, noisy_3, noisy_4,
    noisy_5, noisy_6, noisy_7, noisy_8, noisy_9
]

testNeuralNet(X_testing,weightsMatrix,biasVector)