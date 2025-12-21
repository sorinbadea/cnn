"""
verdict module, issue the final verdict based on analyzer results
"""

def check_for_100_shape(eucl_results, shape_with_max_eucl):
    """
    check if a unique shape has 100 % euclidian confidence
    @param eucl_results - hash of euclidian results/shape
    @param shape_with_max_eucl - shape name which maximum euclidian confidence
    """
    shapes_with_100_eucl_results = [key for key in eucl_results if round(eucl_results[key]*100, 2) == 100.0]
    if len(shapes_with_100_eucl_results) == 1 and shapes_with_100_eucl_results[0] == shape_with_max_eucl:
        return shape_with_max_eucl

    return None

def verdict(cosine_result, eucl_result):
    """
    issue the final verdict following the evaluation results:
    - if cosine evaluation is low for all shapes, return "Unknown pattern"
    - if one shape has 100 % euclidian confidence, ignore cosine evaluation
    - if two shapes have high euclidian confidence ( >= 83.33 ) take the biggest cosine evaluation
    - if max euclidian result has 66.66 % and the cosine evaluation matches, take that shape
    - if max euclidian result has 66.66 % but the cosine evaluation does not match:
        check the euclidian result for the cosine shape:
         - if below 33.33 % take the euclidian match
         - if above 50.00 % take the cosine evaluation
    - if max euclidian result has 33.33 % and the cosine evaluation matches, take that shape
    @param cosine_result - hash with cosine similarity/shape
    @param eucl_result; hash of euclidian matches/shape
    """
    cosine_match = max(cosine_result, key=cosine_result.get)
    eucl_shape_match = max(eucl_result, key=eucl_result.get)
    max_eucl_percent = round(eucl_result[eucl_shape_match] * 100, 2)

    low_eucl_confidence  = round((2/6)*100, 2) # 33.33 %
    mid_eucl_confidence  = round((3/6)*100, 2) # 50.00 %
    good_eucl_confidence = round((4/6)*100, 2) # 66.66 %
    high_eucl_confidence = round((5/6)*100, 2) # 83.33 %

    result = "Unknown pattern"

    # do not consider any results if cosine evaluation is low
    if max(cosine_result.values()) < 5.999:
        return result

    __shape = check_for_100_shape(eucl_result, eucl_shape_match)
    if __shape != None:
        # only one shape has 100 % euclidian confidence, ignore cosine evaluation
        result = __shape

    elif max_eucl_percent >= good_eucl_confidence and cosine_match == eucl_shape_match:
        # ideal case, both evaluation methods matches
        result = cosine_match

    elif max_eucl_percent >= good_eucl_confidence and cosine_match != eucl_shape_match:
        # cosine evaluation does not match, check other shapes vs cosine match

        if round(eucl_result[cosine_match]*100, 2) < low_eucl_confidence:
            # consider euclidian match if euclidian result for cosine evaluation is not important
            result = eucl_shape_match

        elif round(eucl_result[cosine_match]*100, 2) >= mid_eucl_confidence:
            # consider cosine match if euclidian is still important
            result = cosine_match

    elif max_eucl_percent >= low_eucl_confidence and cosine_match == eucl_shape_match:
        # average  euclidian confidence; cosine evaluation matches
        result = cosine_match

    return result

