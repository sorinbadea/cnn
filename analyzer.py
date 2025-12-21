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

class Euclidian:
    """
    class implementing euclidian distance evaluation
    between trained data and new input pooled data
    """
    def __init__(self, training_data, input_pooled) -> None:
        self._trained_filter = training_data
        self._input_pooled = input_pooled
        self._euclidean_distance = lambda vec1, vec2: np.linalg.norm(vec1 - vec2)

    def __enter__(self) -> 'Euclidian':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)-> None:
        del self._trained_filter
        del self._input_pooled

    def iterate_in_samples(self) -> tuple:
        """
        Generator to iterate through trained samples and input pooled data
        """
        for _pool_row in self._input_pooled:
            for _train_row in self._trained_filter:
                if _pool_row is not None and _train_row is not None:
                    yield _pool_row, _train_row

    def evaluate_euclidian(self) -> tuple:
        """
        Evaluates a single pooled filter result against trained
        patterns using euclidean distance; returns a touple of
        matches and not matches
        @param trained_filter: list of trained samples for a kernel
        @param input_pooled: new input samples to evaluate
        """
        _matches = 0
        _iterations = 0
        for _pooled_row, _trained_row in self.iterate_in_samples():
            _iterations += 1
            _distance = self._euclidean_distance(_trained_row, _pooled_row)
            _nv_average = np.sum(np.array(_pooled_row))/len(_pooled_row)
            _matches += int(_distance < _nv_average * 0.09)
            #TODO, add a fast euclidian match if _matches >= 2:
            #    break

        return _matches, _iterations - _matches

def evaluate(pooled_maps, shape, db, verbose=False) -> dict:
    """
    Returns a dict of two maps:
    a map containing the euclidian evaluation confidence per shape
    a map containing the cosine max similarities per shape
    @param pooled_maps: map of kernel shapes to pooled outputs
    @param shape: a shape dictionary from filters.py
    @param db: database interface to get trained data
    @param verbose: verbose mode
    """
    _euclidian_result = {}
    _cosine_result = {}
    for key in pooled_maps:
        """
        get the trained pooled maps for each filter
        """
        _trained_filter = db.get_trained_data(key)
        _trained_filter = np.array(_trained_filter)
        # evaluate euclidian distance and cosine similarity
        # -------------------------------------------------
        with Euclidian(_trained_filter, pooled_maps[key]) as eucl:
            _euclidian_result[key] = eucl.evaluate_euclidian()
        _cosine_result[key] = evaluate_cosine(_trained_filter, pooled_maps[key])

    if verbose:
        print(f"analyse result for shape '{shape['name']}'")
        print("========================================")
        # display the euclidian evaluation
        print(_euclidian_result)
        # display the cosine evaluation
        display_cosine_result(_cosine_result)

    ## evaluate the cosine results
    _cosine_eval  = 0
    for key in _cosine_result.keys():
        _similarity = []
        for c in _cosine_result[key]:
            _similarity.append(c)
        _cosine_eval += max(_similarity)
    if verbose:
        print("total of cosine kernel similarities", _cosine_eval)

    ## evaluate the euclidian results
    _total_matches = sum(1 for m in [ _euclidian_result[key][0] for key in _euclidian_result ] if m > 0)
    _result = {}
    _result['euclidian'] = _total_matches / len(_euclidian_result) if len(_euclidian_result) > 0 else 0
    _result['cosine'] = _cosine_eval
    return _result

# the more random changes comparing the trained data, the more
# low confidence, even if the values are 10x higher than trained data,
# if follows the trend of trained data, than is OK
# trained_pool = [[10, 11, 12, 13], [11, 12, 13, 13], [11.5, 12, 13, 12.5]]
# new_pool = [[1, 12, 3, 3] , [11, 12, 13, 13], [15, 36, 9, 10], [111, 122, 130, 130 ]]
# result = evaluate_cosine(new_pool, np.array(trained_pool), "test")
