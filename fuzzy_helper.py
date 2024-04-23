import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_fuzzy_rank(output_score):
    # Complement of Gaussian density function
    fuzzy_rank = 1 - np.exp(-output_score)
    return fuzzy_rank


def calculate_fuzzy_ranks(final_outputs):
    fuzzy_ranks = [calculate_fuzzy_rank(score) for score in final_outputs]
    return fuzzy_ranks


def calculate_confidence_factors(fuzzy_rank):
    confidence_factor_ranks = [(rank) for rank in fuzzy_rank]
    # confidence_factor_ranks = [(1) for rank in fuzzy_rank]

    return confidence_factor_ranks


def find_sum_fuzzy_ranks(fuzzy_ranks):
    # Initialize a result list with zeros
    result = [0] * len(fuzzy_ranks[0]) if fuzzy_ranks else []

    # Iterate over each sublist and update the sums
    for sublist in fuzzy_ranks:
        for i, num in enumerate(sublist):
            result[i] = max(result[i], num)
            # result[i] += num

    return result


def multiply_fuzzy_sum_confidence_factors(fuzzy,confidence):
    result = [fuzzy[i]*confidence[i] for i in range(0,len(fuzzy))]
    return result


def normalise(arr):
    scaler = StandardScaler()
    normalized_arr = scaler.fit_transform(arr)
    return normalized_arr


def get_best_class(arr):
    # Get the indexes that would sort the array
    sorted_indexes = np.argsort(arr)
    rank_array = np.empty_like(sorted_indexes)
    rank_array[sorted_indexes] = np.arange(len(arr))
    return np.argmin(rank_array)
