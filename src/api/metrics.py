from prometheus_client import Counter, Histogram, Info

# Requests counter

""" How many requests have been made to the API, labeled by endpoint and method. """
REQUESTS_COUNTER = Counter(
    "request_count",
    "Total HTTP requests to the API, labeled by endpoint, method, and status code",
    ["endpoint", "method", "status_code"]
)

# Request latency histogram

""" How fast we responding """
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Histogram of request latency (seconds) for the API, labeled by endpoint and method",
    ["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)

# prediction counter

"""what is the distribution of predicted emotions (positive vs negative)"""
PREDICTION_COUNTER = Counter(
    "prediction_count",
    "Total number of predictions made by the API, labeled by emotion",
    ["emotion"]
)

# Prediction confidence histogram

"""What is the distribution of confidence scores for the predictions made by the API, labeled by emotion"""
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Histogram of confidence scores for predictions made by the API, labeled by emotion",
    ["emotion"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Model info

"""Information about the model, such as version and type."""
MODEL_INFO = Info(
    "model_info",
    "Information about the model, such as version and type"
)