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
    eucl_percent = round(eucl_result[eucl_dist_match] * 100)

    if eucl_percent > 66 and cosine_match == eucl_dist_match:
        # ideal case, both evaluation methods matches
        # evaluate other shape confidence
        print(cosine_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    elif eucl_percent > 70 and cosine_match != eucl_dist_match:
        # evaluate other shape confidence
        for key in eucl_result:
            if cosine_match == key and (eucl_result[key]*100) >= 40:
                 #consider cosine match if euclidian is still important
                 print(cosine_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)
                 return
        # higher euclidian match but not matching the cosine, take eculidian
        print(eucl_dist_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    elif eucl_percent > 30 and eucl_percent < 71 and cosine_match == eucl_dist_match:
        # low euclidian confidence, take cosine
        print(eucl_dist_match, " with euclidian distance confidence of", eucl_percent, "% and cosine evaluation", cosine_match)

    else:
        # unknown pattern
        print("unknow patern, low euclidian confidence", eucl_percent, "% cosine evaluation", cosine_match)