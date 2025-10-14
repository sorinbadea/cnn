import numpy as np
import filters
from database import DataBaseInterface

"""
Module to evaluate pooled outputs against trained patterns
using euclidean distance
"""

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

def is_match_distance(new_values, trained_values):
    """Check if new values match within distance threshold"""
    distance = euclidean_distance(new_values, trained_values)
    # try to evaluate an appropiate threshold
    nv_average = np.sum(np.array(new_values))/len(new_values)
    threshold = nv_average * 0.1
    return distance < threshold, distance

def iterate_in_samples(trained_filter, input_pooled):
    """
    Generator to iterate through trained samples and input pooled data
    """
    for pool_row in input_pooled:
        for train_row in trained_filter:
            yield pool_row, train_row

def evaluate_filter(trained_filter, input_pooled, verbose=False):
    """
    Evaluates a single pooled filter result against trained 
    patterns using euclidean distance;
    @param trained_filter: list of trained samples for a kernel
    @param input_pooled: new input samples to evaluate
    """
    matches = 0
    not_matches = 0
    for pool_row, train_row in iterate_in_samples(trained_filter, input_pooled):
        match, distance = is_match_distance(pool_row, train_row)
        if match:
            matches += 1
        else:
            not_matches += 1
    return matches, not_matches

def evaluate(pooled_maps, verbose=False):
    """
    Returns a map with the result of evaluation for each kernel;
    @param pooled_maps: map of kernel shapes to pooled outputs@
    """
    result = {}
    db = DataBaseInterface('localhost','myapp','postgres','password',5432)
    for key in pooled_maps:
        trained_filter = db.get_data(key)
        if trained_filter is False:
            print(f"❌ Error fetching data for filter '{key}'")
            return None
        else:
            result[key] = evaluate_filter(trained_filter, pooled_maps[key], verbose)
    db.database_disconnect()
    matches = [ result[key][0] for key in result ]
    total_matches = sum(1 for m in matches if m > 0)
    return total_matches/len(result)

if __name__ == "__main__":
    # Usage, is_match_distance function
    trained_pool = [0.8, 0.3, 0.6, 0.9]
    new_pool = [0.82, 0.28, 0.61, 0.88]
    match, dist = is_match_distance(new_pool, trained_pool, threshold=0.1)
    print(f"Match: {match}, Distance: {dist:.4f}")