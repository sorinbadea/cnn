import numpy as np
import math
import filters

"""
Module to evaluate pooled outputs against trained patterns
implements cosine evaluation and euclidian distance evaliation
"""

"""
using cosine
"""

def cosine_similarity(trained_data, new_data):
    """
    evaluare cosine similarity new data vs traoined data
    @param: trained data; data already processed
    @param new_data : new data obtained via convolution
    """
    # helpers
    def inner_prod(X, Y):
        sum_prod = 0
        for x_i, y_i in zip(X, Y):
            sum_prod += x_i * y_i
        return sum_prod

    def magnitude(X):
        sum_squares = 0
        for x_i in X:
            sum_squares += x_i ** 2
        return math.sqrt(sum_squares)

    mag_X = magnitude(new_data)
    mag_Y = magnitude(trained_data)
    # Check if either magnitude is zero or very close to zero
    if mag_X == 0 or mag_Y == 0:
        return 0.0
    return inner_prod(new_data, trained_data) / (mag_X * mag_Y)

def get_similarity(trained_data, new_data):
    """
    returns the max similarity for the given input and trained data
    @param new_data : new data obtained via convolution
    @param: trained data; data already processed
    """
    results = []
    for trained_row in trained_data:
        similarity = cosine_similarity(trained_row, new_data)
        results.append((trained_row, similarity))
    # Sort by similarity in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    # results [1] is highest similarity among the whole set
    return results[1]

def evaluate_cosine(trained_data, new_data, verbose=False):
    results = []
    t_data = [ t_row[0] for t_row in list(trained_data) ]
    if verbose:
        print("pooled_data=", new_data.tolist())
        print("trained_data=", t_data)
    for pool_row in new_data.tolist():
        results.append(get_similarity(t_data, pool_row))
    return results

def display_cosine_result(output):
    """
    show the max similarity from output
    @param output: cosine similarity for each applied kernel, (filter)
    """
    for key in output:
        ## debug print("")
        ## debug print("key", key)
        similarity = []
        for c in output[key]:
            similarity.append(c[1])
        print(f"Cosine max similarity '{max(similarity)}' for kernel '{key}'")

"""
using euclidean distance
"""

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

def is_match_distance(trained_values, new_values):
    """Check if new values match within distance threshold"""
    distance = euclidean_distance(new_values, trained_values)
    # try to evaluate an appropiate threshold
    nv_average = np.sum(np.array(new_values))/len(new_values)
    threshold = nv_average * 0.09
    return distance < threshold, distance

def iterate_in_samples(trained_filter, input_pooled):
    """
    Generator to iterate through trained samples and input pooled data
    """
    for pool_row in input_pooled:
        for train_row in trained_filter:
            yield pool_row, train_row

def evaluate_euclidian(trained_filter, input_pooled):
    """
    Evaluates a single pooled filter result against trained
    patterns using euclidean distance; returns a triple of
    matches, not matches and euclidian distance
    @param trained_filter: list of trained samples for a kernel
    @param input_pooled: new input samples to evaluate
    """
    matches = 0
    not_matches = 0
    min_distance = 255.0
    for pooled_row, trained_row in iterate_in_samples(trained_filter, input_pooled):
        match, distance = is_match_distance(trained_row, pooled_row)
        matches += int(match)
        not_matches += int(not match)
        # evaluate the minial distance for this kernel
        if min_distance > round(distance.item(), 2):
            min_distance = round(distance.item(), 2)
    return matches, not_matches, min_distance

def evaluate(pooled_maps, shape_index, db, verbose=False):
    """
    Returns a map with the result of evaluation for each kernel;
    @param pooled_maps: map of kernel shapes to pooled outputs
    @param shape_index: shape index
    """
    euclidian_result = {}
    cosine_result = {}
    for key in pooled_maps:
        trained_filter = db.get_trained_data(key)
        if trained_filter is False:
            print(f"❌ Error fetching data for filter '{key}'")
            return None
        else:
            euclidian_result[key] = evaluate_euclidian(trained_filter, pooled_maps[key])
            cosine_result[key] = evaluate_cosine(trained_filter, pooled_maps[key])
    if verbose:
        print(f"analyse result for shape '{filters.shapes[shape_index]['name']}'")
        print(euclidian_result)
        display_cosine_result(cosine_result)

    ## evaluate the euclidian results
    total_matches = sum(1 for m in [ euclidian_result[key][0] for key in euclidian_result ] if m > 0)

    ## evaluate the cosine results
    kernel_similarity  = 0
    for key in cosine_result:
        similarity = []
        for c in cosine_result[key]:
            similarity.append(c[1])
        kernel_similarity += max(similarity)
    if verbose:
        print("average of kernel similarities", kernel_similarity)
    return total_matches/len(euclidian_result), kernel_similarity

if __name__ == "__main__":
    # the more random changes comparing the trained data, the more
    # low confidence, even if the values are 10x higher than trained data,
    # if follows the trend of trained data, than is OK
    trained_pool = [[10, 11, 12, 13], [11, 12, 13, 13], [11.5, 12, 13, 12.5]]
    new_pool = [[1, 12, 3, 3] , [11, 12, 13, 13], [15, 36, 9, 10], [111, 122, 130, 130 ]]
    result = evaluate_cosine(new_pool, trained_pool, "test")
    for i in result:
        #print(i[0])
        print(i[1])
