"""fivefu: shared library for the 5-FU resistance analysis.

The scripts under scripts/ are the pipeline stages; this package holds the code
they share. The task is regression: `AUC` always means the GDSC dose-response
area under the curve, never ROC-AUC.
"""

__version__ = "1.0.0"
