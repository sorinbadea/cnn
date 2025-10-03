import numpy as np
import filters
from database import DataBaseInterface

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

def is_match_distance(new_values, trained_values, threshold=0.5):
    """Check if new values match within distance threshold"""
    distance = euclidean_distance(new_values, trained_values)
    return distance < threshold, distance

class PooledMatcher:
    def __init__(self):
        self.trained_patterns = []
        self.threshold = None
        
    def train(self, positive_samples, negative_samples=None):
        """
        Calculate adaptive threshold from training samples
        
        Args:
            positive_samples: List of similar pooled outputs
            negative_samples: List of different pooled outputs (optional)
        """
        self.trained_patterns = positive_samples
        
        # Calculate intra-class distances
        intra_distances = []
        for i, p1 in enumerate(positive_samples):
            for p2 in positive_samples[i+1:]:
                intra_distances.append(euclidean_distance(p1, p2))
        
        if intra_distances:
            # Set threshold as mean + 2*std of intra-class distances
            mean_dist = np.mean(intra_distances)
            std_dist = np.std(intra_distances)
            self.threshold = mean_dist + 2 * std_dist
        else:
            self.threshold = 0.5  # Default
        print(f"Adaptive threshold set to: {self.threshold:.4f}")
    
    def match(self, new_values):
        """Check if new values match any trained pattern"""
        if not self.trained_patterns:
            raise ValueError("Must train matcher first")
            
        distances = [euclidean_distance(new_values, pattern) 
                    for pattern in self.trained_patterns]
        min_dist = min(distances)
        return min_dist < self.threshold, min_dist

def evaluate_filter(trained_filter, pooled):
    """
    Evaluate a single pooled filter against trained patterns
    @param trained_filter: list of trained pooled outputs from database
    @param pooled_filter: new pooled output to evaluate"""
    matcher = PooledMatcher()
    for row in trained_filter:
        for item in row:
            print("Trained with:", item, " length:", len(item))
            matcher.train(item)
    print("---- input pooled data ----")
    print(pooled)
    is_match, distance = matcher.match(pooled)
    print(f"Match: {is_match}, Distance: {distance:.4f}")

def evaluate(pooled_maps):
    """
    Evaluate pooled maps using PooledMatcher
    @param pooled_maps: list of pooled outputs to evaluate
    """
    db = DataBaseInterface('localhost','myapp','postgres','password',5432)
    for key, pooled_filter in zip (filters.kernels_digit_one['filters'], pooled_maps):
        print("")
        print("filter:", key)
        print("")
        trained_filter = db.get_data(key)
        if trained_filter:
            evaluate_filter(trained_filter, pooled_filter)
        else:
            print(f"❌ Cannot fetch data for filter '{key}'")
    db.database_disconnect()       

if __name__ == "__main__":
    # Usage
    matcher = PooledMatcher()
    # Train with similar examples
    similar_outputs = [
        [0.8, 0.3, 0.6, 0.9],
        [0.82, 0.28, 0.61, 0.88],
        [0.79, 0.31, 0.59, 0.91],
    ]
    matcher.train(similar_outputs)

    # Test new input
    test_output = [[0.81, 0.30, 0.60, 0.89]]
    is_match, distance = matcher.match(test_output)
    print(f"Match: {is_match}, Distance: {distance:.4f}")

    # Usage, is_match_distance function
    trained_pool = [0.8, 0.3, 0.6, 0.9]
    new_pool = [0.82, 0.28, 0.61, 0.88]
    match, dist = is_match_distance(new_pool, trained_pool, threshold=0.1)
    print(f"Match: {match}, Distance: {dist:.4f}")