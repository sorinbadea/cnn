"""
verdict module, issue the final verdict based on analyzer results
"""
def check_other_shapes(eucl_results, shape_of_100, treshold):
    """
    check if other shapes have euclidian distance over treshold
    @param eucl_results - hash of euclidian results/shape
    @param shape_of_100 - shape name which has 100 % euclidian confidence
    @param treshold - euclidian distance treshold"""
    shapes_with_100 = [key for key in eucl_results if (eucl_results[key]*100) == 100]
    if len(shapes_with_100) == 1:
        for key in eucl_results:
            if key == shape_of_100:
                continue
            if (round(eucl_results[key]*100)) > treshold:
                return False
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
    eucl_dist_match = max(eucl_result, key=eucl_result.get)
    max_eucl_percent = round(eucl_result[eucl_dist_match] * 100)

    low_eucl_confidence  = round((2/7)*100) # 28.57 %
    min_eucl_confidence  = round((3/7)*100) # 42.85 %
    med_eucl_confidence  = round((4/7)*100) # 57.14 %
    high_eucl_confidence = round((5/7)*100) # 71.42 %

    if check_other_shapes(eucl_result, eucl_dist_match, med_eucl_confidence):
        # only one shape has 100 % euclidian confidence, ignore cosine evaluation
        print(eucl_dist_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)

    elif max_eucl_percent > med_eucl_confidence and cosine_match == eucl_dist_match:
        # ideal case, both evaluation methods matches
        print(cosine_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)

    elif max_eucl_percent > med_eucl_confidence and cosine_match != eucl_dist_match:
        # euclidian confidence is average and cosine evaluation
        # does not match, check other shapes with cosine match
        for key in eucl_result:
            if cosine_match == key and (round(eucl_result[key]*100)) >= min_eucl_confidence:
                 #consider cosine match if euclidian is still important
                 print(cosine_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)
                 return
    
        # good euclidian match but not matching the cosine, take eculidian
        print(eucl_dist_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)

    elif max_eucl_percent > low_eucl_confidence and max_eucl_percent < high_eucl_confidence and cosine_match == eucl_dist_match:
        # average  euclidian confidence; cosine evaluation matches
        print(cosine_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)

    elif max_eucl_percent > low_eucl_confidence and max_eucl_percent < high_eucl_confidence and cosine_match != eucl_dist_match:
        # good euclidian evaluation but not matching cosine
        for key in eucl_result:
            if cosine_match == key and (round(eucl_result[key]*100)) >= min_eucl_confidence:
                 #consider cosine match if euclidian is still important
                 print(cosine_match, " with euclidian distance confidence of", max_eucl_percent, "% and cosine evaluation", cosine_match)
                 return
    else:
        # unknown pattern
        print("unknow patern, low euclidian confidence", max_eucl_percent, "% cosine evaluation", cosine_match)
