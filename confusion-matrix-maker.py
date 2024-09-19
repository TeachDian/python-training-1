import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Number of classes
num_classes = 11
max_sum = 140  # Maximum sum per row/column

# Initialize confusion matrix
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Define the focus area around [6,6] and assign higher values to surrounding cells
focus_value = np.random.randint(40, 50)
confusion_matrix[6, 6] = focus_value  # Main focus

# Distribute higher values around [6,6]
confusion_matrix[5, 5] = np.random.randint(30, 40)
confusion_matrix[5, 6] = np.random.randint(20, 30)
confusion_matrix[5, 7] = np.random.randint(15, 25)
confusion_matrix[6, 5] = np.random.randint(20, 30)
confusion_matrix[6, 7] = np.random.randint(15, 25)
confusion_matrix[7, 5] = np.random.randint(15, 25)
confusion_matrix[7, 6] = np.random.randint(20, 30)
confusion_matrix[7, 7] = np.random.randint(30, 40)

# Fill in the rest of the matrix with random values and more 0s, ensuring the row/column sum <= 140
for i in range(num_classes):
    remaining_row = max_sum - np.sum(confusion_matrix[i, :])  # Update remaining row sum
    if remaining_row > 0:
        for j in range(num_classes):
            if confusion_matrix[i, j] == 0 and remaining_row > 0:  # Only fill empty cells
                # Increase the probability of 0 values being generated
                if np.random.rand() < 0.6:  # 60% chance to assign 0
                    value = 0
                else:
                    # For row or column 10, assign lower values (0 to 7) unless it's the focus area
                    if i == 10 or j == 10:
                        value = np.random.randint(0, 8)  # Smaller values for row and column 10
                    else:
                        value = np.random.randint(0, min(remaining_row + 1, 20))  # Keep values lower elsewhere
                confusion_matrix[i, j] = value
                remaining_row -= value

    # Ensure the sum of each row is 140 or less
    remaining_row = max_sum - np.sum(confusion_matrix[i, :])
    if remaining_row > 0:
        confusion_matrix[i, -1] += remaining_row

# Adjust values in the last column (column 10) to be between 0 and 30
for i in range(num_classes):
    if confusion_matrix[i, 10] > 30:
        confusion_matrix[i, 10] = np.random.randint(0, 31)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=range(num_classes))

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
plt.show()
