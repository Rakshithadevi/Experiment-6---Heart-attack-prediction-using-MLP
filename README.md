# Experiment-6---Heart-attack-prediction-using-MLP
## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python
## Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<br>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<br>

## Program:
```
Name:Rakshitha Devi J
Reg No:212221230082
```
```

import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv("/content/heart.csv")
X=data.iloc[:, :-1].values #features 
Y=data.iloc[:, -1].values  #labels 

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss=mlp.fit(X_train,y_train).loss_curve_

y_pred=mlp.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Losss")
plt.show()

```



## Output:
### X Values:

![image](https://github.com/Rakshithadevi/Experiment-6---Heart-attack-prediction-using-MLP/assets/94165326/49ef8f34-1adb-472a-8738-b32ef451d153)

### Y Values:

![image](https://github.com/Rakshithadevi/Experiment-6---Heart-attack-prediction-using-MLP/assets/94165326/75cde7d4-7e33-4ccd-815a-08f28e0688be)

### Accuracy:

![image](https://github.com/Rakshithadevi/Experiment-6---Heart-attack-prediction-using-MLP/assets/94165326/3d8a1ef0-bf65-4ede-98d5-1b3a2d29a941)

### Loss Convergence Graph:

![image](https://github.com/Rakshithadevi/Experiment-6---Heart-attack-prediction-using-MLP/assets/94165326/11875740-4b45-449c-a460-e55b25c837df)



## Result:

Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     

