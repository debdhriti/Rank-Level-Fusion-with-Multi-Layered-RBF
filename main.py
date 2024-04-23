import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import csv
import matplotlib
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import rbf
import rbf_helper
import random
import warnings
import fuzzy_helper
import svm
import extract_data

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


X_train, Y_train, X_test, Y_test, size_of_tr, data = extract_data.extractIris(25)


# Example usage:
# Assume X is your input data
# X = ...

# Set the number of clusters and nearest neighbors

# X_train = np.array([
#     [1, 2],
#     [3, 4],
#     [5, 6],
#     [7,8],
#     [9,10],
#     [3,2],
#     [7,1],
#     [9,4],
#     [6,5]
# ])






# Plotting the points
# plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', label='Data Points')

# # Adding labels and title
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter Plot of Data Points')

# # Display the legend
# plt.legend()

# Show the plot
# plt.show()

xp, yp = [],[]


num_clusters = [size_of_tr]
# for i in range(int(size_of_tr//10) + 10):
#   if size_of_tr - i*10 <= 0:
#     break
#   num_clusters.append(size_of_tr - i*10);

# print(num_clusters)
num_neighbors = 10
num_outputs = 3

# # Get receptors and spreads
# receptors, spreads = rbf_input_middle_layers(X_train, num_clusters, num_neighbors)

# # Print the results
# print("Receptors:")
# print(receptors)
# print("Spreads:")
# print(spreads)

# Have a training set and
# step1 -> have the middle layer ready(DONE)
# step2 -> train the weights, output the weights matrix
# iterate over the training set

# Y_train = np.array([
#     0,
#     0,
#     1,
#     1,
#     1,
#     1,
#     0,
#     1,
#     1
# ])


print(X_train)
print(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(r)
# print(s)


iter = size_of_tr
cnt = 1
rbf_dm = []
old_acc = 0
while iter > 0:
  print("FOR THE HIDDEN LAYER CONFIGURATION: ", num_clusters)
  X_train_trans = X_train
  r, s =[], []
  pm = []

  for i in range(len(num_clusters)):
        r_curr, s_curr = rbf_helper.rbf_input_middle_layers(np.array(X_train_trans), num_clusters[i], num_neighbors)
        pm = rbf.get_phi_matrix(X_train_trans, r_curr, s_curr)
        r.append(r_curr)
        s.append(s_curr)
        X_train_trans = pm

  wm = rbf.train_output_layer_weights(X_train_trans, num_outputs, Y_train)

  # print(wm)
  tmp_rbf = []
  acc = rbf.check_accurary(X_test, Y_test, r, s, num_clusters, num_outputs, wm, tmp_rbf)
  if acc > old_acc:
    rbf_dm = tmp_rbf
    old_acc = acc

  print(acc)

  xp.append(cnt)
  yp.append(acc)

  #updation
  iter -= 10
  cnt += 1
  num_clusters.append(iter)


# print(X_train)
# print(X_test)

# print(r)
# print(s)
# print(wm)


print(xp)
print(yp)


# plt.figure(figsize=(8,6))
# plt.plot(xp,yp,color='blue', marker='o', linestyle='-', label='Line')
# plt.title("Accuracies plotted against the added hidden layers:")
# plt.xlabel("Hidden Layers")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()

# Plotting the line graph
# plt.figure(figsize=(8, 6))  # Adjust the size of the plot if needed
# plt.plot(xp, yp, color='blue', marker='o', linestyle='-', label='Line')  # Line plot of points
# plt.title('Line Graph of Points')  # Set title for the plot
# plt.xlabel('X-axis')  # Label for x-axis
# plt.ylabel('Y-axis')  # Label for y-axis
# plt.legend()  # Show legend
# plt.grid(True)  # Show grid
# plt.show()  # Display the plot


decision_scores, y_test = svm.evaluateSVM(data, X_train, X_test, Y_train, Y_test)
print("rbf ka dams matrix: ", fuzzy_helper.normalise(rbf_dm))



# DONO DAMS MATRICES MIL GAYE
# AB KIMATU PIPELINE

def twodify(arr):
    return [[x] for x in arr]

def onedify(arr):
  return [x[0] for x in arr]

def merge_models(output_score1, output_score2):
  output_score1 = fuzzy_helper.normalise(output_score1)
  output_score2 = fuzzy_helper.normalise(output_score2)
  rf, wf = 0, 0
  for i in range(len(output_score1)):
    fuzzy_rank1, fuzzy_rank2 = fuzzy_helper.calculate_fuzzy_ranks(output_score1[i]), fuzzy_helper.calculate_fuzzy_ranks(output_score2[i])
    cf1, cf2 = fuzzy_helper.calculate_confidence_factors(fuzzy_rank1), fuzzy_helper.calculate_confidence_factors(fuzzy_rank2)
    fuzzy_result = fuzzy_helper.find_sum_fuzzy_ranks([fuzzy_rank1, fuzzy_rank2])
    normalised_cf = onedify(fuzzy_helper.normalise(twodify(fuzzy_helper.find_sum_fuzzy_ranks([cf1, cf2]))))
    predicted_classes = fuzzy_helper.multiply_fuzzy_sum_confidence_factors(fuzzy_result, normalised_cf)
    if np.argmax(predicted_classes) == y_test[i]:
      rf += 1
    else:
      wf += 1
  print("fuzzy accuracy: ", (rf / (rf + wf)) * 100)


merge_models(decision_scores, rbf_dm)