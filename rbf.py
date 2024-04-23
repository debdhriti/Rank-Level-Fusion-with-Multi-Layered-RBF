import rbf_helper
import numpy as np


def get_phi_matrix(X, receptor_list, spread_list):
    phi_matrix = []
    for i in range(len(X)):
        curr = []
        for j in range(len(receptor_list)):
            curr.append(rbf_helper.calculate_activation(X[i], receptor_list[j], spread_list[j]))
        phi_matrix.append(curr)
    return phi_matrix



def train_output_layer_weights(phi_matrix, num_outputs, Y_train):
    # row =
    weight_matrix = []
    phi_pseudo_inv = np.linalg.pinv(phi_matrix)
    # print(phi_pseudo_inv)
    for i in range(num_outputs):
        b = []
        for j in Y_train:
            if j[i] == 1:
                b.append([1])
            else:
                b.append([0])
        output_weights = np.dot(phi_pseudo_inv, b)
        weight_matrix.append(output_weights)
    return weight_matrix


def check_accurary(X_test, Y_test, receptor_list, spread_list, num_clusters, num_outputs, weight_matrix, tmp_rbf):
    right, wrong = 0, 0
    for i in range(len(X_test)):
        middle_layer_output, curr_middle_layer_output = [], X_test[i]
        for j in range(len(num_clusters)):
          middle_layer_output = []
          for k in range(len(receptor_list[j])):
              middle_layer_output.append(rbf_helper.calculate_activation(curr_middle_layer_output, receptor_list[j][k], spread_list[j][k]))
          # print(i, " " , j, " ", middle_layer_output)
          curr_middle_layer_output = middle_layer_output


        final_output = []
        for j in range(num_outputs):
            val = 0
            for k in range(len(curr_middle_layer_output)):
                val = val + (weight_matrix[j][k][0])*curr_middle_layer_output[k]
            final_output.append(val)

        tmp_rbf.append(final_output)
        if np.argmax(Y_test[i]) == np.argmax(final_output):
            right += 1
        else:
            wrong += 1
    print("right: ", right)
    print("wrong: ", wrong)
    return float(right / (right + wrong)) * 100


