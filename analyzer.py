"""
Module to evaluate pooled outputs against trained patterns
implements cosine evaluation and euclidian distance evaliation
"""
import numpy as np
"""
using cosine
"""

def cosine_similarity(trained_data, new_data):
    """
    evaluare cosine similarity new data vs trained data
    @param: trained data; data already processed
    @param new_data : new data obtained via convolution
    """
    vectA = np.array(new_data)
    vectB = np.array(trained_data)
    # Calculate norms
    norm_A = np.linalg.norm(vectA)
    norm_B = np.linalg.norm(vectB)
    # Check if either magnitude is zero or very close to zero
    if norm_A == 0.0 or norm_B == 0.0:
        return 0.0
    return np.dot(vectA, vectB) / (norm_A * norm_B)

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
    # results [0][1] is highest similarity among the whole set
    return results[0][1]

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
        #print("")
        #print("key", key)
        similarity = []
        for cos in output[key]:
            similarity.append(cos)
        print(f"Cosine max similarity '{max(similarity)}' for kernel '{key}'")

"""
using euclidean distance
"""

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

def is_match_distance(trained_values, new_values):
    """ Check if new values match within distance threshold """
    distance = euclidean_distance(trained_values, new_values)
    nv_average = np.sum(np.array(new_values))/len(new_values)
    threshold = nv_average * 0.09
    return distance < threshold

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
    patterns using euclidean distance; returns a touple of
    matches and not matches
    @param trained_filter: list of trained samples for a kernel
    @param input_pooled: new input samples to evaluate
    """
    matches = 0
    not_matches = 0
    for pooled_row, trained_row in iterate_in_samples(trained_filter, input_pooled):
        match = is_match_distance(trained_row, pooled_row)
        matches += int(match)
        not_matches += int(not match)
    return matches, not_matches

def evaluate(pooled_maps, shape, db, verbose=False):
    """
    Returns the euclidian evaluation confidence (percent) and 
    the total cosine similarity for the shape index
    @param pooled_maps: map of kernel shapes to pooled outputs
    @param shape: a shape dictionary from filters.py
    @param db: database interface to get trained data
    @param verbose: verbose mode
    """
    euclidian_result = {}
    cosine_result = {}
    for key in pooled_maps:
        """
        get the trained pooled maps for each filter
        """
        trained_filter = db.get_trained_data(key)
        # evaluate euclidian distance and cosine similarity
        # -------------------------------------------------
        euclidian_result[key] = evaluate_euclidian(trained_filter, pooled_maps[key])
        cosine_result[key] = evaluate_cosine(trained_filter, pooled_maps[key])

    if verbose:
        print(f"analyse result for shape '{shape['name']}'")
        print("========================================")
        # display the euclidian evaluation
        print(euclidian_result)
        # display the cosine evaluation
        display_cosine_result(cosine_result)

    ## evaluate the cosine results
    cosine_eval  = 0
    for key in cosine_result.keys():
        similarity = []
        for c in cosine_result[key]:
            similarity.append(c)
        cosine_eval += max(similarity)
    if verbose:
        print("total of cosine kernel similarities", cosine_eval)

    ## evaluate the euclidian results
    total_matches = sum(1 for m in [ euclidian_result[key][0] for key in euclidian_result ] if m > 0)
    result = {}
    result['euclidian'] = total_matches / len(euclidian_result) if len(euclidian_result) > 0 else 0
    result['cosine'] = cosine_eval
    return result

# the more random changes comparing the trained data, the more
# low confidence, even if the values are 10x higher than trained data,
# if follows the trend of trained data, than is OK
# trained_pool = [[10, 11, 12, 13], [11, 12, 13, 13], [11.5, 12, 13, 12.5]]
# new_pool = [[1, 12, 3, 3] , [11, 12, 13, 13], [15, 36, 9, 10], [111, 122, 130, 130 ]]
# result = evaluate_cosine(new_pool, np.array(trained_pool), "test")
