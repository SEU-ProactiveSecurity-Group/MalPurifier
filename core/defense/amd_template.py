"""
Abstract class representing a detector for adversarial malware.
"""


class DetectorTemplate(object):
    def __init__(self):
        self.tau = None                     # Threshold variable
        # Flag indicating whether the detector is enabled
        self.is_detector_enabled = True

    def forward(self, x):
        """
        Class prediction and density estimation
        """
        raise NotImplementedError

    def get_threshold(self):
        """
        Compute the threshold for rejecting outliers
        """
        raise NotImplementedError

    def get_tau_sample_wise(self):
        """
        Get the tau value for each sample
        """
        raise NotImplementedError

    def indicator(self):
        """
        Return a boolean flag vector indicating whether to reject a sample
        """
        raise NotImplementedError
