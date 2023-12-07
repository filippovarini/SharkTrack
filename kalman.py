from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

def initialize_kalman_filter(bbox):
    """
    Initialise a new Kalman Filter for each detected shark. Each Kalman filter
    is an id of the shark.

    Parameters
    ----------
    bbox : list
        List of bounding box coordinates for the detected shark.
        bbox[0], bbox[1] = x, y coordinates of the centre of the bounding box.
    """
    # Initial state: [x, y, velocity_x, velocity_y]
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([bbox[0], bbox[1], 0., 0.])
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],  # x = x + v_x * dt
                     [0, 0, 1, 0],  # y = y + v_y * dt
                     [0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0],  # Measurement function, extract x, y from state
                     [0, 1, 0, 0]])

    kf.R *= np.array([[0.1, 0],       # Measurement noise
                      [0, 0.1]])
    
    kf.P *= 10.                     # Initial uncertainty

    kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.05)  # Process noise

    return kf


def match_detection_to_track(kalman_filters, detection):
    """
    Given a new shark detection, matches it with an existing filter. 
    TODO: : this doesn't take into account new sharks or sharks that have left 
    the frame.

    Parameters
    ----------
    kalman_filters : list
        List of Kalman filters.
    detection : list
        List of bounding box coordinates for the detected shark.
        detection[0], detection[1] = x, y coordinates of the centre of the 
        bounding box.
    """
    # Simple nearest neighbor matching
    distances = []
    for kf in kalman_filters:
        predicted_position = kf.x[:2]
        distance = np.linalg.norm(predicted_position - detection)
        distances.append(distance)
    
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return min_index 
