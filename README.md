# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Screenshot 2025-03-03 114655](https://github.com/user-attachments/assets/302773ed-6ccf-42d7-9340-193d6e3dbb2c)




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: TARUN S
### Register Number: 212223040226
```python
# Name:TARUN S
# Register Number:212223040226
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,4)
        self.fc2=nn.Linear(4,6)
        self.fc3=nn.Linear(6,1)
        self.relu= nn.ReLU()
        self.history ={'Loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion= nn.MSELoss()
optimizer= optim.RMSprop(ai_brain.parameters() ,lr=0.001)


# Name:Tarun S
# Register Number:212223040226
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=3000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['Loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

![Screenshot 2025-03-10 100244](https://github.com/user-attachments/assets/dbf7920b-a997-420c-96af-9b6b16e4d656)



## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/71723a8d-4144-4ef4-88e3-c59e28a56904)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/dbfb80cd-1efb-4794-a508-e54a306d6869)




## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
