"""Turbine Health Monitoring Module.

Layered exactly like the Failure Prediction Module so an engineer moving
between them finds the same shapes:

``config``
    Which configuration files this module needs merged together.
``sensor_rules``
    Physical limits and operating-envelope rules, loaded from configuration and
    evaluated against a frame.
``regimes``
    Operating-regime assignment (idle / low / medium / high load / curtailed).
``health_features``
    The single leakage-safe feature entry point, used by training and serving.
``drift``
    CUSUM, EWMA and Isolation Forest sensor-drift detection.
``health_score``
    Target construction, candidate training, selection and regression metrics.
``health_class``
    Score-to-class banding and the drift penalty that links the two.
``persistence``
    Health-artifact paths, saving and loading.
``narratives``
    Grounded explanation and advisory text.

Importing this package has no side effects: nothing reads artifacts, trains or
touches the data directories at import time.
"""
