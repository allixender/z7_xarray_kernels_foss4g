import numpy as np


class HexGridDistortionModel:
    def __init__(self, empirical_data: dict):
        """
        Initializes the model by pre-computing the scaling weights 
        for every region to ensure lightning-fast lookups later.
        """
        # The exact geometric constant derived earlier (~0.9523)
        self.base_factor = np.sqrt(np.pi / (2 * np.sqrt(3)))
        
        # This will hold our pre-calculated weights
        self.region_weights = {}
        
        # Pre-compute weights on startup
        for region_id, measurements in empirical_data.items():
            geom_mean = measurements['mean_d_geom']
            
            self.region_weights[region_id] = {
                'w_k': measurements['mean_d_k'] / geom_mean,
                'w_j': measurements['mean_d_j'] / geom_mean,
                'w_i': measurements['mean_d_i'] / geom_mean
            }

    def get_neighbor_distances(self, region_key: str, cls: float) -> dict:
        """
        Returns the actual anisotropic distances to neighbors for a given cls.
        """
        # 1. Calculate the theoretical perfectly-regular distance
        base_d = cls * self.base_factor
        
        # 2. Fetch the pre-computed stretch/squeeze weights
        weights = self.region_weights.get(region_key)
        
        if not weights:
            # Fallback: if region isn't in lookup, assume a perfect hexagon
            return {'d_k': base_d, 'd_j': base_d, 'd_i': base_d}
            
        # 3. Apply the weights to the base distance
        return {
            'd_k': base_d * weights['w_k'],
            'd_j': base_d * weights['w_j'],
            'd_i': base_d * weights['w_i']
        }

