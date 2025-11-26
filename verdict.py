"""
verdict module, issue the final verdict based on analyzer results
"""

def check_100_shapes(eucl_results, shape_of_100):
    """
    check if a unique shape has 100 % euclidian confidence
    @param eucl_results - hash of euclidian results/shape
    @param shape_of_100 - shape name which has 100 % euclidian confidence
    """
    shapes_with_100 = [key for key in eucl_results if round(eucl_results[key]*100, 2) == 100.0]
    if len(shapes_with_100) == 1 and shapes_with_100[0] == shape_of_100:
        return True
    return False

def verdict(cosine_result, eucl_result):
    """
    issue the final verdict
    for euclidian distance, get the maximum match from all tried filters
    for cosine similarity get the shape with the maximum similarities
    @param cosine_result - name of the shape with the best cosine similarity
    @param eucl_result; hash of euclidian matches/shape
    """
    cosine_match = max(cosine_result, key=cosine_result.get)
    eucl_shape_match = max(eucl_result, key=eucl_result.get)
    max_eucl_percent = round(eucl_result[eucl_shape_match] * 100, 2)

    low_eucl_confidence  = round((2/7)*100, 2) # 28.57 %
    min_eucl_confidence  = round((3/7)*100, 2) # 42.85 %
    med_eucl_confidence  = round((4/7)*100, 2) # 57.14 %
    high_eucl_confidence = round((5/7)*100, 2) # 71.42 %

    result = "Unknown pattern"

    if check_100_shapes(eucl_result, eucl_shape_match):
        # only one shape has 100 % euclidian confidence, ignore cosine evaluation
        result = eucl_shape_match
        #print(eucl_shape_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)

    elif max_eucl_percent >= med_eucl_confidence and cosine_match == eucl_shape_match:
        # ideal case, both evaluation methods matches
        result = cosine_match

    elif max_eucl_percent >= med_eucl_confidence and cosine_match != eucl_shape_match:
        # cosine evaluation does not match, check other shapes vs cosine match

        if round(eucl_result[cosine_match]*100, 2) <= low_eucl_confidence:
            # consider euclidian match if euclidian result for cosine evaluation is not important
            result = eucl_shape_match

        elif round(eucl_result[cosine_match]*100, 2) >= min_eucl_confidence:
            # consider cosine match if euclidian is still important
            result = cosine_match

    elif max_eucl_percent >= low_eucl_confidence and max_eucl_percent <= high_eucl_confidence and cosine_match == eucl_shape_match:
        # average  euclidian confidence; cosine evaluation matches
        result = cosine_match

    return result

