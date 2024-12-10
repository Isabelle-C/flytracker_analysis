import numpy as np


class CoordinateTransform:
    @staticmethod
    def wrap2pi(ang: np.array) -> np.array:
        """
        Wrap a set of values to fit between zero and 2Pi.
        """

        positiveValues = ang > 0
        wrappedAng = ang % (2 * np.pi)
        # Handle edge case where angle is 0
        wrappedAng[(ang == 0) & positiveValues] = 2 * np.pi

        return wrappedAng

    @staticmethod
    def cart2pol(x, y):
        """
        Returns radius * theta in radians

        Arguments:
            x -- _description_
            y -- _description_
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)
