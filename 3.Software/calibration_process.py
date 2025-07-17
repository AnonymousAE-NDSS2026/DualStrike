import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import numpy as np
from classify import KeypressClassifier

# =====================================
# GLOBAL CONFIGURATION PARAMETERS
# =====================================

# Data Filtering Parameters
FILTER_TYPE = 'savgol'          # Type of filter to apply: 'savgol' or 'moving_average'
FILTER_WINDOW = 15              # Window size for filtering operations
SAVGOL_POLY_ORDER = 3          # Polynomial order for Savitzky-Golay filter

# Offset Calculation Parameters
OFFSET_WINDOW_SIZE = 25         # Window size for calculating sensor offset values
OFFSET_STD_THRESHOLD = 3        # Standard deviation threshold for detecting stable segments

# Envelope Calculation Parameters
ENVELOPE_WINDOW_SIZE_SECONDS = 0.05  # Time window in seconds for envelope calculation

# Peak Detection Parameters (Method 1 - Basic)
PEAK_THRESHOLD_STD = 5          # Standard deviation threshold for peak detection
MIN_FLAT_DURATION = 0.05        # Minimum duration in seconds for flat segments

# Peak Detection Parameters (Method 2 - Scipy based)
PROMINENCE_THRESHOLD = 5        # Peak prominence threshold for scipy.find_peaks
WIDTH_THRESHOLD = 0.05          # Minimum peak width in seconds

# Peak Detection Parameters (Method 3 - Slope based, recommended)
SLOPE_THRESHOLD = 0.1           # Slope threshold for detecting rising/falling edges
AMPLITUDE_THRESHOLD = 1.5       # Amplitude threshold above baseline for valid peaks
SLOPE_WINDOW_SIZE = 0.1         # Time window in seconds for slope calculation

# Machine Learning Model Parameters
MODEL_PATH = 'wooting_keypress_model2.pth'  # Path to the trained classification model
OUTPUT_CSV_FILENAME = 'wooting_predictions_dy2cm.csv'  # Output file for predictions

# Visualization Parameters
FIGURE_SIZE_LARGE = (20, 24)    # Figure size for multi-sensor plots
FIGURE_SIZE_MEDIUM = (15, 8)    # Figure size for combined plots
FIGURE_SIZE_SMALL = (12, 8)     # Figure size for single plots
PLOT_ALPHA_MAIN = 0.7          # Alpha value for main plot lines
PLOT_ALPHA_BACKGROUND = 0.5     # Alpha value for background data
PLOT_ALPHA_HIGHLIGHT = 0.2      # Alpha value for highlighted regions

# Data Processing Constants
NUM_SENSORS = 8                 # Total number of magnetic field sensors
NUM_AXES = 3                   # Number of axes per sensor (X, Y, Z)
SAMPLING_RATE_ESTIMATE = 100    # Estimated sampling rate in Hz
BASE_PERCENTILE = 10           # Percentile used for baseline level calculation
PLATEAU_THRESHOLD_RATIO = 0.9   # Ratio for determining peak plateau region

# File Processing Parameters
INPUT_CSV_PATH = r'wooting_align_dy2cm.csv'  # Input CSV file path

# =====================================
# END OF CONFIGURATION PARAMETERS
# =====================================

class KeypressPredictor:
    """
    A machine learning-based keypress classifier that uses magnetic field data
    from multiple sensors to predict which key was pressed.
    
    This class loads a pre-trained neural network model and provides methods
    to predict key presses based on magnetic field sensor readings.
    """
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the keypress predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved PyTorch model file
        """
        # Load model and label encoder from checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize the neural network model
        self.model = KeypressClassifier(checkpoint['num_classes']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load the label encoder for converting predictions back to key names
        self.label_encoder = checkpoint['label_encoder']
    
    def predict(self, peak_data):
        """
        Predict the key type based on magnetic field peak data.
        
        This method takes magnetic field readings from all sensors at a peak moment
        and returns the most likely key that was pressed along with the confidence.
        
        Args:
            peak_data (numpy.ndarray): Shape (8, 3) array containing magnetic field
                                     readings from 8 sensors with X, Y, Z components
        
        Returns:
            tuple: (predicted_label, probability)
                - predicted_label (str): The predicted key name
                - probability (float): Confidence probability (0-1)
        
        Raises:
            ValueError: If input data shape is not (8, 3)
        """
        # Validate input data format
        if peak_data.shape != (NUM_SENSORS, NUM_AXES):
            raise ValueError(f"Input data shape must be ({NUM_SENSORS}, {NUM_AXES}), "
                           f"but got {peak_data.shape}")
        
        # Reshape data to match model input format (1, 1, 8, 3)
        data = torch.FloatTensor(peak_data.reshape(1, 1, NUM_SENSORS, NUM_AXES)).to(self.device)
        
        # Perform prediction using the neural network
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the highest probability prediction
            prob, predicted = torch.max(probabilities, 1)
            predicted_label = self.label_encoder.inverse_transform([predicted.item()])[0]
            
            return predicted_label, prob.item()

def read_magnetic_data(file_path, filter_type=FILTER_TYPE, window=FILTER_WINDOW, 
                      poly_order=SAVGOL_POLY_ORDER):
    """
    Read and filter magnetic field data from CSV file for all sensors.
    
    This function reads raw magnetic field data from a CSV file containing
    readings from multiple sensors, applies filtering to reduce noise,
    and returns processed data for each sensor.
    
    Args:
        file_path (str): Path to the CSV file containing magnetic field data
        filter_type (str): Type of filter to apply ('savgol' or 'moving_average')
        window (int): Window size for filtering operations
        poly_order (int): Polynomial order for Savitzky-Golay filter
    
    Returns:
        dict: Dictionary containing filtered data for each sensor
              Key format: 'sensor_N' where N is sensor number (1-8)
              Value: DataFrame with columns ['Time(s)', 'Magnetic Field X', 'Magnetic Field Y', 'Magnetic Field Z']
    """
    df = pd.read_csv(file_path)
    
    # Calculate relative time in seconds from the first timestamp
    df['Time(s)'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Dictionary to store processed data for all sensors
    all_sensors_data = {}
    
    # Process magnetic field data for each of the 8 sensors
    for sensor_num in range(1, NUM_SENSORS + 1):
        filtered_data = {}
        
        # Apply filtering to each axis (X, Y, Z) of the current sensor
        for axis in ['x', 'y', 'z']:
            raw_data = df[f'sensor_{sensor_num}_{axis}']
            
            if filter_type == 'savgol':
                # Apply Savitzky-Golay filter for smooth noise reduction
                filtered_data[axis] = savgol_filter(raw_data, window, poly_order)
            elif filter_type == 'moving_average':
                # Apply moving average filter
                filtered_data[axis] = raw_data.rolling(window=window, center=True).mean()
                # Fill NaN values that result from windowing
                filtered_data[axis] = filtered_data[axis].fillna(method='bfill').fillna(method='ffill')
        
        # Create DataFrame for current sensor with filtered data
        sensor_data = pd.DataFrame({
            'Time(s)': df['Time(s)'],
            'Magnetic Field X': filtered_data['x'],
            'Magnetic Field Y': filtered_data['y'],
            'Magnetic Field Z': filtered_data['z']
        })
        
        all_sensors_data[f'sensor_{sensor_num}'] = sensor_data
    
    print(f"Applied {filter_type} filtering with window size: {window}")
    print(f"Successfully read data from {NUM_SENSORS} magnetic field sensors")
    return all_sensors_data

def calculate_offset(data, window_size=OFFSET_WINDOW_SIZE, std_threshold=OFFSET_STD_THRESHOLD):
    """
    Calculate sensor offset values by finding stable baseline periods.
    
    This function identifies stable periods in the magnetic field data where
    the sensor readings have low variance, indicating no key press activity.
    These periods are used to calculate baseline offset values for calibration.
    
    Args:
        data (pd.DataFrame): Sensor data with magnetic field measurements
        window_size (int): Number of data points to analyze for stability
        std_threshold (float): Standard deviation threshold for detecting stable periods
    
    Returns:
        dict: Offset values for each axis {'X': offset_x, 'Y': offset_y, 'Z': offset_z}
    
    Note:
        The std_threshold has been adjusted to accommodate the new data range.
    """
    offset = {'X': 0, 'Y': 0, 'Z': 0}
    
    for axis in ['X', 'Y', 'Z']:
        column = f'Magnetic Field {axis}'
        
        # Search for the first stable period by analyzing consecutive windows
        for i in range(len(data) - window_size):
            window = data[column][i:i+window_size]
            window_std = window.std()
            print(window_std)
            
            # If the standard deviation is below threshold, we found a stable period
            if window_std < std_threshold:
                offset[axis] = window.mean()
                break
        else:
            # If no stable period is found, use the initial data points
            print(f"Warning: No stable period found for {axis} axis, "
                  f"using mean of first {window_size} points")
            offset[axis] = data[column][:window_size].mean()
    
    return offset

def calculate_magnetic_field(data, offset):
    """
    Convert raw sensor readings to calibrated magnetic field values in microTesla.
    
    This function applies offset correction and gain calibration to convert
    raw sensor readings into physically meaningful magnetic field measurements.
    
    Args:
        data (pd.DataFrame): Raw sensor data
        offset (dict): Offset values for each axis calculated from stable periods
    
    Returns:
        pd.DataFrame: Data with additional columns for calibrated magnetic field values
                     - 'MagX (uT)': X-axis magnetic field in microTesla
                     - 'MagY (uT)': Y-axis magnetic field in microTesla  
                     - 'MagZ (uT)': Z-axis magnetic field in microTesla
                     - 'Total Field (uT)': Magnitude of total magnetic field vector
    """
    # Gain factor for converting to microTesla (currently set to 1.0)
    gain = 1.0
    
    # Apply offset correction and gain calibration
    data['MagX (uT)'] = (data['Magnetic Field X'] - offset['X']) / gain
    data['MagY (uT)'] = (data['Magnetic Field Y'] - offset['Y']) / gain
    data['MagZ (uT)'] = (data['Magnetic Field Z'] - offset['Z']) / gain
    
    # Calculate the magnitude of the total magnetic field vector
    data['Total Field (uT)'] = np.sqrt(
        data['MagX (uT)']**2 + 
        data['MagY (uT)']**2 + 
        data['MagZ (uT)']**2
    )
    return data

def plot_magnetic_field(data):
    """
    Plot magnetic field data showing all three axes and total field magnitude.
    
    This function creates a comprehensive plot showing the magnetic field
    components over time, useful for visualizing sensor behavior and key press events.
    
    Args:
        data (pd.DataFrame): Magnetic field data with time and field components
    """
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    plt.plot(data['Time(s)'], data['MagX (uT)'], label='X axis')
    plt.plot(data['Time(s)'], data['MagY (uT)'], label='Y axis')
    plt.plot(data['Time(s)'], data['MagZ (uT)'], label='Z axis')
    plt.plot(data['Time(s)'], data['Total Field (uT)'], label='Total Field', linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (μT)')
    plt.title('Magnetic Field vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_envelope(data, window_size_seconds=ENVELOPE_WINDOW_SIZE_SECONDS):
    """
    Calculate the envelope of magnetic field signals for peak detection.
    
    The envelope represents the outer boundary of the signal variations,
    which is useful for detecting key press events that cause rapid changes
    in the magnetic field readings.
    
    Args:
        data (pd.DataFrame): Magnetic field data with calibrated measurements
        window_size_seconds (float): Time window in seconds for envelope calculation
    
    Returns:
        pd.DataFrame: Envelope data with columns:
                     - 'Time(s)': Time values
                     - 'Envelope_X', 'Envelope_Y', 'Envelope_Z': Axis envelopes
                     - 'Envelope_Total': Total field envelope
    """
    # Convert time window to number of data points (assuming ~100 Hz sampling)
    window_points = int(window_size_seconds * SAMPLING_RATE_ESTIMATE)
    
    envelope = pd.DataFrame()
    envelope['Time(s)'] = data['Time(s)']
    
    # Calculate envelope for each magnetic field axis
    for axis in ['X', 'Y', 'Z']:
        column = f'Mag{axis} (uT)'
        signal = data[column].values
        envelope_values = np.zeros_like(signal)
        
        # For each point, find the maximum absolute value in its local window
        for i in range(len(signal)):
            start_idx = max(0, i - window_points//2)
            end_idx = min(len(signal), i + window_points//2)
            window = signal[start_idx:end_idx]
            
            # Find the value with maximum absolute magnitude in the window
            max_abs_idx = np.argmax(np.abs(window))
            envelope_values[i] = window[max_abs_idx]
        
        envelope[f'Envelope_{axis}'] = envelope_values
    
    # Calculate envelope for total field magnitude
    total_field = data['Total Field (uT)'].values
    total_envelope = np.zeros_like(total_field)
    
    for i in range(len(total_field)):
        start_idx = max(0, i - window_points//2)
        end_idx = min(len(total_field), i + window_points//2)
        window = total_field[start_idx:end_idx]
        max_abs_idx = np.argmax(np.abs(window))
        total_envelope[i] = window[max_abs_idx]
    
    envelope['Envelope_Total'] = total_envelope
    
    return envelope

def plot_magnetic_field_with_envelope(data, envelope):
    """
    Plot magnetic field data with its envelope lines.
    
    This function creates a two-panel plot showing both the raw magnetic field
    data and the calculated envelope lines for visualization and analysis.
    
    Args:
        data (pd.DataFrame): Magnetic field data with calibrated measurements
        envelope (pd.DataFrame): Envelope data with calculated envelope values
    """
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    
    # Plot raw magnetic field data
    plt.subplot(211)
    plt.plot(data['Time(s)'], data['MagX (uT)'], label='X-axis', alpha=PLOT_ALPHA_BACKGROUND)
    plt.plot(data['Time(s)'], data['MagY (uT)'], label='Y-axis', alpha=PLOT_ALPHA_BACKGROUND)
    plt.plot(data['Time(s)'], data['MagZ (uT)'], label='Z-axis', alpha=PLOT_ALPHA_BACKGROUND)
    plt.plot(data['Time(s)'], data['Total Field (uT)'], label='Total Field', alpha=PLOT_ALPHA_BACKGROUND)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (μT)')
    plt.title('Raw Magnetic Field Data')
    plt.grid(True)
    plt.legend()
    
    # Plot envelope lines
    plt.subplot(212)
    plt.plot(envelope['Time(s)'], envelope['Envelope_X'], label='X-axis Envelope')
    plt.plot(envelope['Time(s)'], envelope['Envelope_Y'], label='Y-axis Envelope')
    plt.plot(envelope['Time(s)'], envelope['Envelope_Z'], label='Z-axis Envelope')
    plt.plot(envelope['Time(s)'], envelope['Envelope_Total'], label='Total Field Envelope', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (μT)')
    plt.title('Magnetic Field Data Envelope')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def detect_peaks_and_flats(envelope, threshold_std=PEAK_THRESHOLD_STD, min_flat_duration=MIN_FLAT_DURATION):
    """
    Detect flat segments and peak segments, merging rising edges, middle flat regions,
    and falling edges into a single complete peak.
    
    This function identifies segments of the signal that represent either
    a flat region (no significant change) or a peak (rapid change).
    
    Args:
        envelope (pd.DataFrame): Envelope data with 'Time(s)' and 'Envelope_Total' columns
        threshold_std (float): Standard deviation threshold for detecting flat segments
        min_flat_duration (float): Minimum duration in seconds for a flat segment
    
    Returns:
        list: List of tuples (start_time, end_time, segment_type)
              - segment_type is 'peak' or 'flat'
    """
    signal = envelope['Envelope_Total'].values
    times = envelope['Time(s)'].values
    base_level = np.percentile(signal, BASE_PERCENTILE)  # Use 10% percentile as baseline
    
    # Use sliding window to calculate local standard deviation
    window_points = int(min_flat_duration * SAMPLING_RATE_ESTIMATE)
    local_std = np.array([np.std(signal[max(0, i-window_points//2):min(len(signal), i+window_points//2)])
                         for i in range(len(signal))])
    
    # Initialize segments
    segments = []
    in_peak = False
    peak_start = 0
    last_level = 'base'  # 'base', 'rising', 'high', 'falling'
    
    for i in range(1, len(signal)):
        current_value = signal[i]
        is_active = current_value > base_level + threshold_std
        
        if not in_peak and is_active:
            # Start a new peak
            in_peak = True
            peak_start = i
            last_level = 'rising'
        elif in_peak:
            if last_level == 'rising':
                if local_std[i] < threshold_std:
                    last_level = 'high'
            elif last_level == 'high':
                if local_std[i] > threshold_std and signal[i] < signal[i-1]:
                    last_level = 'falling'
            elif last_level == 'falling':
                if not is_active:
                    # End current peak
                    segments.append((times[peak_start], times[i], 'peak'))
                    in_peak = False
                    last_level = 'base'
    
    # Handle the last possible peak
    if in_peak:
        segments.append((times[peak_start], times[-1], 'peak'))
    
    # Add flat segments
    flat_segments = []
    last_end = times[0]
    
    for start, end, _ in segments:
        if start > last_end:
            flat_segments.append((last_end, start, 'flat'))
        last_end = end
    
    if last_end < times[-1]:
        flat_segments.append((last_end, times[-1], 'flat'))
    
    # Combine and sort all segments
    all_segments = segments + flat_segments
    all_segments.sort(key=lambda x: x[0])
    
    return all_segments

def detect_peaks_and_flats_v3(envelope, slope_threshold=SLOPE_THRESHOLD, amplitude_threshold=AMPLITUDE_THRESHOLD, window_size=SLOPE_WINDOW_SIZE):
    """
    Peak detection based on slope method.
    
    This function identifies peaks in the magnetic field signal by analyzing
    the slope of the envelope and the amplitude relative to the baseline.
    
    Args:
        envelope (pd.DataFrame): Envelope data with 'Time(s)' and 'Envelope_Total' columns
        slope_threshold (float): Slope threshold for detecting rising/falling edges
        amplitude_threshold (float): Amplitude threshold above baseline for valid peaks
        window_size (float): Time window in seconds for slope calculation
    
    Returns:
        list: List of tuples (start_time, end_time, segment_type)
              - segment_type is 'peak' or 'flat'
    """
    signal = envelope['Envelope_Total'].values
    times = envelope['Time(s)'].values
    fs = 1 / np.mean(np.diff(times))  # Calculate sampling rate
    window_points = int(window_size * fs)
    
    # Calculate baseline level
    base_level = np.percentile(signal, BASE_PERCENTILE)
    
    # Calculate slope
    def calculate_slope(data, window):
        slopes = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window//2)
            end_idx = min(len(data), i + window//2)
            if end_idx - start_idx > 1:
                slopes[i] = np.polyfit(range(end_idx-start_idx), 
                                     data[start_idx:end_idx], 1)[0]
        return slopes
    
    slopes = calculate_slope(signal, window_points)
    
    # Initialize segments
    segments = []
    in_peak = False
    peak_start = 0
    
    # State machine variables
    rising_count = 0
    falling_count = 0
    min_count = int(0.05 * fs)  # Minimum duration (50ms)
    
    for i in range(1, len(signal)):
        current_slope = slopes[i]
        current_value = signal[i]
        
        # Determine state
        is_rising = current_slope > slope_threshold
        is_falling = current_slope < -slope_threshold
        is_above_threshold = current_value > base_level + amplitude_threshold
        
        if not in_peak:
            if is_rising and is_above_threshold:
                rising_count += 1
                if rising_count >= min_count:
                    # Confirm start of rising
                    in_peak = True
                    peak_start = i - rising_count
                    rising_count = 0
            else:
                rising_count = 0
        else:  # In peak
            if is_falling:
                falling_count += 1
                if falling_count >= min_count and current_value < base_level + amplitude_threshold:
                    # Confirm end of falling
                    segments.append((times[peak_start], times[i], 'peak'))
                    in_peak = False
                    falling_count = 0
            elif not is_above_threshold:
                # Directly return to baseline level
                if i - peak_start > min_count:
                    segments.append((times[peak_start], times[i], 'peak'))
                in_peak = False
                falling_count = 0
    
    # Handle the last possible peak
    if in_peak:
        segments.append((times[peak_start], times[-1], 'peak'))
    
    # Add flat segments
    flat_segments = []
    last_end = times[0]
    
    for start, end, _ in segments:
        if start > last_end:
            flat_segments.append((last_end, start, 'flat'))
        last_end = end
    
    if last_end < times[-1]:
        flat_segments.append((last_end, times[-1], 'flat'))
    
    # Combine and sort all segments
    all_segments = segments + flat_segments
    all_segments.sort(key=lambda x: x[0])
    
    return all_segments

def detect_peaks_and_flats_v2(envelope, prominence_threshold=PROMINENCE_THRESHOLD, width_threshold=WIDTH_THRESHOLD):
    """
    Peak detection based on peak feature method.
    
    This function identifies peaks in the magnetic field signal using
    the scipy.find_peaks function, which finds local maxima.
    
    Args:
        envelope (pd.DataFrame): Envelope data with 'Time(s)' and 'Envelope_Total' columns
        prominence_threshold (float): Peak prominence threshold
        width_threshold (float): Minimum peak width in seconds
    
    Returns:
        list: List of tuples (start_time, end_time, segment_type)
              - segment_type is 'peak' or 'flat'
    """
    from scipy.signal import find_peaks
    
    signal = envelope['Envelope_Total'].values
    times = envelope['Time(s)'].values
    fs = 1 / np.mean(np.diff(times))  # Calculate sampling rate
    
    # Use scipy's find_peaks function to find peaks
    peaks, properties = find_peaks(signal, 
                                 prominence=prominence_threshold,
                                 width=width_threshold*fs,
                                 rel_height=0.5)
    
    # Initialize segments
    segments = []
    
    # Determine peak segments based on peak width
    for i, peak in enumerate(peaks):
        left_idx = int(peak - properties['widths'][i])
        right_idx = int(peak + properties['widths'][i])
        
        # Ensure indices are within valid range
        left_idx = max(0, left_idx)
        right_idx = min(len(times)-1, right_idx)
        
        segments.append((times[left_idx], times[right_idx], 'peak'))
    
    # Add flat segments
    flat_segments = []
    last_end = times[0]
    
    for start, end, _ in sorted(segments):
        if start > last_end:
            flat_segments.append((last_end, start, 'flat'))
        last_end = end
    
    if last_end < times[-1]:
        flat_segments.append((last_end, times[-1], 'flat'))
    
    # Combine and sort all segments
    all_segments = segments + flat_segments
    all_segments.sort(key=lambda x: x[0])
    
    return all_segments

def calculate_peak_vectors(data, envelope, segments):
    """
    Calculate feature vectors for each peak segment.
    
    This function extracts features from the magnetic field data within
    each identified peak segment, including the mean values of the
    previous flat segment and the plateau region within the peak.
    
    Args:
        data (pd.DataFrame): Magnetic field data with calibrated measurements
        envelope (pd.DataFrame): Envelope data with 'Time(s)' and 'Envelope_Total' columns
        segments (list): List of tuples (start_time, end_time, segment_type)
    
    Returns:
        list: List of dictionaries containing peak information
    """
    peak_vectors = []
    
    # Find all peak and flat segments
    peak_segments = [seg for seg in segments if seg[2] == 'peak']
    flat_segments = [seg for seg in segments if seg[2] == 'flat']
    
    for i, (start_time, end_time, _) in enumerate(peak_segments):
        # Find the previous flat segment
        prev_flat = None
        for flat in flat_segments:
            if flat[1] <= start_time:  # Find the closest flat segment before the peak
                prev_flat = flat
        
        if prev_flat is None:
            continue
            
        # Calculate mean values for the previous flat segment
        flat_mask = (data['Time(s)'] >= prev_flat[0]) & (data['Time(s)'] <= prev_flat[1])
        flat_means = {
            'X': data.loc[flat_mask, 'MagX (uT)'].mean(),
            'Y': data.loc[flat_mask, 'MagY (uT)'].mean(),
            'Z': data.loc[flat_mask, 'MagZ (uT)'].mean()
        }
        
        # Find the high plateau within the peak segment
        peak_signal = envelope.loc[(envelope['Time(s)'] >= start_time) & 
                                 (envelope['Time(s)'] <= end_time), 'Envelope_Total']
        peak_max = peak_signal.max()
        plateau_threshold = peak_max * PLATEAU_THRESHOLD_RATIO  # Consider values above 90% of max as plateau
        
        plateau_mask = (data['Time(s)'] >= start_time) & \
                      (data['Time(s)'] <= end_time) & \
                      (envelope['Envelope_Total'] >= plateau_threshold)
        
        # Calculate peak vector using only the plateau region
        peak_means = {
            'X': data.loc[plateau_mask, 'MagX (uT)'].mean() - flat_means['X'],
            'Y': data.loc[plateau_mask, 'MagY (uT)'].mean() - flat_means['Y'],
            'Z': data.loc[plateau_mask, 'MagZ (uT)'].mean() - flat_means['Z']
        }
        
        # Calculate magnitude
        magnitude = np.sqrt(peak_means['X']**2 + peak_means['Y']**2 + peak_means['Z']**2)
        
        peak_vectors.append({
            'peak_number': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'vector': peak_means,
            'magnitude': magnitude
        })
    
    return peak_vectors

def plot_segments_and_envelope(all_data, all_envelopes, all_segments):
    """
    Plot magnetic field data for each sensor.
    
    This function creates a multi-sensor plot showing the three-axis
    magnetic field data and the total envelope for all sensors.
    
    Args:
        all_data (dict): Dictionary containing processed data for all sensors
        all_envelopes (dict): Dictionary containing envelope data for all sensors
        all_segments (dict): Dictionary containing segment data for all sensors
    """
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    
    for i, sensor_num in enumerate(range(1, NUM_SENSORS + 1), 1):  # 8 sensors
        sensor_key = f'sensor_{sensor_num}'
        data = all_data[sensor_key]
        envelope = all_envelopes[sensor_key]
        segments = all_segments[sensor_key]
        
        plt.subplot(4, 2, i)  # 4 rows, 2 columns layout
        plt.plot(data['Time(s)'], data['MagX (uT)'], 'r', label='X-axis', alpha=PLOT_ALPHA_MAIN)
        plt.plot(data['Time(s)'], data['MagY (uT)'], 'g', label='Y-axis', alpha=PLOT_ALPHA_MAIN)
        plt.plot(data['Time(s)'], data['MagZ (uT)'], 'b', label='Z-axis', alpha=PLOT_ALPHA_MAIN)
        plt.plot(envelope['Time(s)'], envelope['Envelope_Total'], 'k', 
                label='Total Envelope', linestyle='--', alpha=PLOT_ALPHA_BACKGROUND)
        
        # Mark segments
        colors = {'flat': 'green', 'peak': 'red'}
        for start_time, end_time, seg_type in segments:
            plt.axvspan(start_time, end_time, alpha=PLOT_ALPHA_HIGHLIGHT, color=colors[seg_type])
        
        plt.xlabel('Time (s)')
        plt.ylabel('Magnetic Field (μT)')
        plt.title(f'Sensor {sensor_num} - Three-axis Magnetic Field Data')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_peak_summary(peak_vectors):
    """
    Print the number of detected peaks.
    
    This function prints the total number of peaks detected across all sensors.
    
    Args:
        peak_vectors (list): List of dictionaries containing peak information
    """
    print(f"Total number of peaks detected: {len(peak_vectors)}")

def merge_overlapping_peaks(all_segments):
    """
    Merge overlapping peaks across all sensors.
    
    This function combines overlapping peak segments from different sensors
    into a single segment, ensuring that the final output represents
    the true key press duration.
    
    Args:
        all_segments (dict): Dictionary containing segment data for all sensors
    
    Returns:
        list: List of tuples (start_time, end_time) representing merged peaks
    """
    merged_peaks = []
    
    # Combine peak segments from all sensors into a single list
    for sensor_segments in all_segments.values():
        for start, end, seg_type in sensor_segments:
            if seg_type == 'peak':
                merged_peaks.append((start, end))
    
    # Sort by start time
    merged_peaks.sort(key=lambda x: x[0])
    
    # Merge overlapping peaks
    combined_peaks = []
    current_start, current_end = merged_peaks[0]
    
    for start, end in merged_peaks[1:]:
        if start <= current_end:  # If there is overlap
            current_end = max(current_end, end)
        else:
            combined_peaks.append((current_start, current_end))
            current_start, current_end = start, end
    
    combined_peaks.append((current_start, current_end))
    
    return combined_peaks

def process_key_presses(file_path):
    """
    Process key press data to return time segments for each key press.
    
    This function reads key press data from a CSV file and returns
    a list of dictionaries containing the key, start time, and end time
    for each key press event.
    
    Args:
        file_path (str): Path to the CSV file containing key press data
    
    Returns:
        pd.DataFrame: DataFrame with columns ['key', 'start', 'end']
    """
    df = pd.read_csv(file_path)
    
    # Calculate relative time in seconds from the first timestamp
    base_time = df['timestamp'].iloc[0]
    df['relative_time'] = df['timestamp'] - base_time
    
    key_data = []
    current_key = None
    start_time = None
    
    # Iterate through each row
    for _, row in df.iterrows():
        if row['key_press'] != 'none' and current_key is None:
            # Start a new key press
            current_key = row['key_press']
            start_time = row['relative_time']
        elif row['key_press'] == 'none' and current_key is not None:
            # Key press ended
            key_data.append({
                'key': current_key,
                'start': start_time,
                'end': row['relative_time']
            })
            current_key = None
    
    # Handle the last possible key press
    if current_key is not None:
        key_data.append({
            'key': current_key,
            'start': start_time,
            'end': df['relative_time'].iloc[-1]
        })
    
    return pd.DataFrame(key_data)

def print_peak_info(sensor_peak_vectors, combined_peaks, key_presses):
    """
    Print peak information for each sensor and combined peaks, including key information.
    
    This function prints the detected peaks for each sensor and the combined
    peaks, including the key information for overlapping key presses.
    
    Args:
        sensor_peak_vectors (dict): Dictionary containing peak information for each sensor
        combined_peaks (list): List of tuples (start_time, end_time) representing merged peaks
        key_presses (pd.DataFrame): DataFrame containing key press data
    
    Returns:
        list: List of tuples (start_time, end_time, normalized_key) for valid combined peaks
    """
    # Print peaks for each sensor
    for sensor_num, peak_vectors in sensor_peak_vectors.items():
        print(f"\nSensor {sensor_num} Peaks:")
        for peak in peak_vectors:
            print(f"  Peak {peak['peak_number']}: {peak['start_time']:.2f}s to {peak['end_time']:.2f}s")
    
    # Print combined peaks, including key information
    print("\nCombined Peaks with Key Information:")
    valid_peaks = []
    for i, (start, end) in enumerate(combined_peaks, 1):
        # Find overlapping key presses
        key_press = key_presses[
            ~((key_presses['end'] < start) | (key_presses['start'] > end))
        ]
        if not key_press.empty:
            original_key = key_press['key'].iloc[0]
            normalized_key = normalize_true_key(original_key)
            key_info = f"(Key: {normalized_key})"
            print(f"  Combined Peak {i}: {start:.2f}s to {end:.2f}s {key_info}")
            valid_peaks.append((start, end, normalized_key))
    
    return valid_peaks

def plot_combined_peaks(all_data, valid_peaks, key_presses):
    """
    Plot combined peaks and highlight key presses.
    
    This function creates a plot showing the total magnetic field magnitude
    for all sensors and highlights the combined peaks and key presses.
    
    Args:
        all_data (dict): Dictionary containing processed data for all sensors
        valid_peaks (list): List of tuples (start_time, end_time, key) for valid combined peaks
        key_presses (pd.DataFrame): DataFrame containing key press data
    """
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    
    # Plot total magnetic field for each sensor
    for sensor_num in range(1, NUM_SENSORS + 1):  # 8 sensors
        sensor_key = f'sensor_{sensor_num}'
        data = all_data[sensor_key]
        plt.plot(data['Time(s)'], data['Total Field (uT)'], 
                label=f'Sensor {sensor_num}', alpha=PLOT_ALPHA_BACKGROUND)
    
    # Get y-axis range for text annotation
    ymin, ymax = plt.ylim()
    text_height = ymax - (ymax - ymin) * 0.1
    
    # Mark combined peaks and key presses
    for start, end, key in valid_peaks:
        # Add peak region shadow
        plt.axvspan(start, end, color='red', alpha=PLOT_ALPHA_HIGHLIGHT)
        
        # Add key annotation at the center of the peak region
        center = (start + end) / 2
        plt.text(center, text_height, 
                f"Key: {key}", 
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.xlabel('Time (s)')
    plt.ylabel('Total Magnetic Field (μT)')
    plt.title('Combined Peaks with Key Press Information')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_peak_max_data(all_data, envelope, start_time, end_time):
    """
    Get magnetic field data for all 8 sensors at the moment of the peak's maximum total field.
    
    This function identifies the moment in time where the total magnetic field
    magnitude is at its maximum within a peak segment and retrieves the
    corresponding magnetic field readings from all sensors.
    
    Args:
        all_data (dict): Dictionary containing processed data for all sensors
        envelope (pd.DataFrame): Envelope data with 'Time(s)' and 'Envelope_Total' columns
        start_time (float): Start time of the peak segment
        end_time (float): End time of the peak segment
    
    Returns:
        numpy.ndarray: Shape (8, 3) array containing magnetic field readings
                       for all 8 sensors at the maximum total field moment.
    """
    # Find the moment of maximum total field within the peak segment
    max_total_field = float('-inf')
    max_time = None
    
    for sensor_num in range(1, NUM_SENSORS + 1):  # 8 sensors
        sensor_key = f'sensor_{sensor_num}'
        data = all_data[sensor_key]
        
        # Get data within the peak segment
        peak_mask = (data['Time(s)'] >= start_time) & (data['Time(s)'] <= end_time)
        peak_data = data[peak_mask]
        
        # Check total field magnitude
        total_field = np.sqrt(
            peak_data['MagX (uT)']**2 + 
            peak_data['MagY (uT)']**2 + 
            peak_data['MagZ (uT)']**2
        )
        current_max = total_field.max()
        
        if current_max > max_total_field:
            max_total_field = current_max
            max_idx = total_field.idxmax()
            max_time = data.loc[max_idx, 'Time(s)']
    
    # Collect data from all 8 sensors at that moment
    sensor_data = np.zeros((NUM_SENSORS, NUM_AXES))
    for sensor_num in range(1, NUM_SENSORS + 1):
        sensor_key = f'sensor_{sensor_num}'
        data = all_data[sensor_key]
        
        # Find the closest time point
        closest_idx = (data['Time(s)'] - max_time).abs().idxmin()
        
        # Add three-axis data
        sensor_data[sensor_num-1] = [
            data.loc[closest_idx, 'MagX (uT)'],
            data.loc[closest_idx, 'MagY (uT)'],
            data.loc[closest_idx, 'MagZ (uT)']
        ]
    
    return sensor_data

def normalize_true_key(true_key):
    """
    Convert true key name to a standardized format.
    
    This function converts various possible key names from the raw data
    into a consistent format that the model can understand.
    
    Args:
        true_key (str): The raw key name from the CSV file
    
    Returns:
        str: The standardized key name
    """
    # List of standard key formats
    standard_keys = ["'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', 
                    'A', 'Alt', 'B', 'C', 'CapsLock', 'Ctrl', 'D', 'E', 'Esc', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'OS', 'P', 'Q', 'R', 'S', 'Shift', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Create a mapping dictionary
    key_mapping = {
        # Alphabet keys - convert to uppercase
        'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
        
        # Various possible formats for special keys
        'alt': 'Alt', 'ALT': 'Alt', 'Alt_L': 'Alt', 'Alt_R': 'Alt',
        'ctrl': 'Ctrl', 'CTRL': 'Ctrl', 'Ctrl_L': 'Ctrl', 'Ctrl_R': 'Ctrl', 'control': 'Ctrl',
        'shift': 'Shift', 'SHIFT': 'Shift', 'Shift_L': 'Shift', 'Shift_R': 'Shift',
        'space': 'Space', 'SPACE': 'Space', 'spacebar': 'Space',
        'escape': 'Esc', 'ESC': 'Esc', 'esc': 'Esc',
        'caps_lock': 'CapsLock', 'capslock': 'CapsLock', 'CAPSLOCK': 'CapsLock',
        'super': 'OS', 'cmd': 'OS', 'windows': 'OS', 'win': 'OS',
        
        # Numeric and symbol keys remain as is
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
        "'": "'", ',': ',', '-': '-', '.': '.', '/': '/', ';': ';'
    }
    
    # If found in mapping, use the mapped value
    if true_key in key_mapping:
        return key_mapping[true_key]
    
    # If it's a single letter, convert to uppercase
    if isinstance(true_key, str) and len(true_key) == 1 and true_key.isalpha():
        return true_key.upper()
    
    # If it's already in standard format, return directly
    if true_key in standard_keys:
        return true_key
    
    # Otherwise, return the original value
    return true_key

def process_peaks_and_predict(all_data, valid_peaks, model_path=MODEL_PATH):
    """
    Process detected peaks and perform predictions using the machine learning model.
    
    This function initializes the KeypressPredictor, iterates through the
    valid combined peaks, and predicts the key pressed at each peak.
    
    Args:
        all_data (dict): Dictionary containing processed data for all sensors
        valid_peaks (list): List of tuples (start_time, end_time, true_key) for valid combined peaks
        model_path (str): Path to the trained classification model file
    
    Returns:
        pd.DataFrame: DataFrame containing the true key, predicted key,
                      probability, start time, and end time for each prediction.
    """
    # Initialize predictor
    try:
        predictor = KeypressPredictor(model_path)
        print(f"Successfully loaded model: {model_path}")
        print(f"Predicting using {NUM_SENSORS} sensors")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return pd.DataFrame()
    
    # Store results
    results = []
    
    for start_time, end_time, true_key in valid_peaks:
        try:
            # Get peak data at the moment of maximum total field
            peak_data = get_peak_max_data(all_data, None, start_time, end_time)
            
            # Perform prediction
            predicted_key, probability = predictor.predict(peak_data)
            
            # Convert true_key to standardized format
            normalized_true_key = normalize_true_key(true_key)
            
            # Store results
            results.append({
                'true_key': normalized_true_key,
                'predicted_key': predicted_key,
                'probability': probability,
                'start_time': start_time,
                'end_time': end_time
            })
            
            print(f"Peak {start_time:.2f}s-{end_time:.2f}s: True={normalized_true_key}, Predicted={predicted_key}, Probability={probability:.3f}")
            
        except Exception as e:
            print(f"Error processing peak {start_time:.2f}s-{end_time:.2f}s: {e}")
            continue
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
        print(f"Results saved to {OUTPUT_CSV_FILENAME}")
        
        # Calculate accuracy
        accuracy = (results_df['true_key'] == results_df['predicted_key']).mean()
        print(f"Prediction Accuracy: {accuracy:.2%}")
    else:
        print("No successful predictions.")
    
    return results_df

def main():
    # Read and process key press data
    key_presses = process_key_presses(INPUT_CSV_PATH)
    
    # Read all sensor data
    all_sensors_data = read_magnetic_data(INPUT_CSV_PATH, 
                                        filter_type=FILTER_TYPE,
                                        window=FILTER_WINDOW,
                                        poly_order=SAVGOL_POLY_ORDER)
    
    # Store processed results for all sensors
    all_processed_data = {}
    all_envelopes = {}
    all_segments = {}
    all_peak_vectors = {}
    
    # Process data for each sensor
    for sensor_num in range(1, NUM_SENSORS + 1):  # 8 sensors
        sensor_key = f'sensor_{sensor_num}'
        data = all_sensors_data[sensor_key]
        
        # Calculate offset and apply
        offset = calculate_offset(data)
        processed_data = calculate_magnetic_field(data, offset)
        
        # Calculate envelope
        envelope = calculate_envelope(processed_data)
        
        # Detect segments
        segments = detect_peaks_and_flats_v3(envelope, 
                                            slope_threshold=SLOPE_THRESHOLD,
                                            amplitude_threshold=AMPLITUDE_THRESHOLD,
                                            window_size=SLOPE_WINDOW_SIZE)
        
        # Calculate peak feature vectors
        peak_vectors = calculate_peak_vectors(processed_data, envelope, segments)
        
        # Store results
        all_processed_data[sensor_key] = processed_data
        all_envelopes[sensor_key] = envelope
        all_segments[sensor_key] = segments
        all_peak_vectors[sensor_key] = peak_vectors
        
        # Print peak count
        print(f"Sensor {sensor_num} - ", end="")
        print_peak_summary(peak_vectors)
    
    # Merge all sensor peaks
    combined_peaks = merge_overlapping_peaks(all_segments)
    
    # Get valid peaks (peaks with key information)
    valid_peaks = print_peak_info(all_peak_vectors, combined_peaks, key_presses)
    
    # Process peaks and perform predictions
    results_df = process_peaks_and_predict(all_processed_data, valid_peaks)
    print("\nPrediction Results:")
    print(results_df)
    
    # Plot combined peaks
    plot_combined_peaks(all_processed_data, valid_peaks, key_presses)
    
    # Plot all sensor charts
    plot_segments_and_envelope(all_processed_data, all_envelopes, all_segments)

if __name__ == "__main__":
    main()