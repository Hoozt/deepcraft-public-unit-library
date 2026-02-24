# Signal Processing Units

## Audio
The 'Audio' directory contains audio-specific signal processing units:

### Spectral
Units used for analyzing the frequency spectrum of audio signals:
- FftShift: Shifts the zero-frequency component to the center of the spectrum
- MelFilterbank: Applies a Mel-scale filterbank to the spectrum
- PowToDb: Converts the power spectrum to a decibel scale
- MelSpectrogram: Generates a Mel spectrogram, representing the spectral content of a signal on the Mel scale

### WindowFunctions
Units that modulate the input signal in the time domain before transforming it into the frequency domain:
- Hamming: Applies the Hamming Window function
- Hann: Applies the Hann Window function

## Transforms
The 'Transforms' directory contains general-purpose units that convert signals from the time domain to the frequency domain:
- Dct: Implements the Discrete Cosine Transform
- Fft: Implements the Fast Fourier Transform (complex input/output)
- RealFft: Implements the Fast Fourier Transform for real numbers only

## Filters
The 'Filters' directory contains basic time-domain filtering operations:
- LowPassFilter: Allows low-frequency signals to pass through while attenuating high frequencies
- HighPassFilter: Allows high-frequency signals to pass through while attenuating low frequencies
- BandPassFilter: Allows frequencies within a specific range to pass through

## MachineLearning
The 'MachineLearning' directory contains units related to machine learning workflows:

### PostProcessing
Post-processing filters for machine learning and computer vision applications:
- ConsecutiveConfidenceFilter: Filters detections based on consecutive confidence thresholds
- BoundingBoxFilter: Filters and processes bounding box detections
- SwarmOutputFilter: Processes swarm intelligence algorithm outputs
- ObjectTracker: Tracks objects across frames
- Threshold: Applies threshold-based filtering

## ImageProcessing
The 'ImageProcessing' directory contains units for image manipulation, drawing, and visualization:

### Manipulation
Units for basic image transformation and manipulation:
- Crop: Extracts a rectangular region from an image
- Resizing: Resizes images to different dimensions
- ScaleImage: Scales images up or down (includes DownscaleImage and UpscaleImage)
- ImagePadding: Adds padding around images

### Drawing
Units for drawing graphics on images:
- DrawBox: Draws rectangular boxes on images
- DrawLine: Draws lines on images
- DrawText: Renders text on images
- BitmapFont: Provides bitmap font support for text rendering

### Visualization
Units for visualizing detection and tracking results:
- DisplayBoundingBox: Visualizes bounding boxes from object detection
- DisplayNumber: Displays numeric values on images
- DisplayObjectTracker: Visualizes object tracking results
- ObjectDetectionCounter: Counts and displays detected objects

## Temporal
The 'Temporal' directory contains units that analyze the time-domain representation of signals:
- SlidingWindow: Applies a sliding window function to the time-domain signal
- ContextualWindow: Provides contextual windowing for temporal analysis

## Radar
The 'Radar' directory contains radar-specific signal processing units:
- CFAR1D: Implements 1D Constant False Alarm Rate detection
- CFAR2D: Implements 2D Constant False Alarm Rate detection

## IMU
The 'IMU' directory contains Inertial Measurement Unit (IMU) and motion-related processing units:
- RotateIMU: Rotates IMU sensor data
- Rotate3DVector: Rotates 3D vectors in space
- SOH: State of Health monitoring for IMU sensors