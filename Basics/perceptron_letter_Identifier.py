weightsVector = [0.0] * 25

matrix_X = [
    1, -1, -1, -1,  1,
   -1,  1, -1,  1, -1,
   -1, -1,  1, -1, -1,
   -1,  1, -1,  1, -1,
    1, -1, -1, -1,  1
]

matrix_T = [
    1,  1,  1,  1,  1,
   -1, -1,  1, -1, -1,
   -1, -1,  1, -1, -1,
   -1, -1,  1, -1, -1,
   -1, -1,  1, -1, -1
]

X_training = [matrix_T, matrix_X]

Y_training = [1, -1]

bias = 0.0

X_testing = matrix_T
Y_expectedOutput = 1

learning_rate = 0.1

iterations = 0

while True:
    error = 0
    for i in range(len(X_training)):
        sample = X_training[i]
        real_output = Y_training[i]
        
        total = 0;
        
        for j in range(len(weightsVector)):
            total += weightsVector[j] * sample[j]
        
        total += bias
        
        activation = 1 if total >= 0 else -1
        
        error = real_output - activation
        
        if(error != 0):
            for j in range(len(weightsVector)):
                weightsVector[j] = weightsVector[j] + learning_rate * error * sample[j]
            bias += learning_rate * error
        
    if(error == 0): break

print("Training completed")
print("Final Weights:")
print(weightsVector)
print("Final bias:")
print(bias)

##TESTING
total = 0;

for j in range(len(weightsVector)):
    total += weightsVector[j] * X_testing[j]
        
total += bias
        
activation = 1 if total >= 0 else -1

print(activation == Y_expectedOutput)