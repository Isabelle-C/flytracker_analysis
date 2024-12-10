import numpy as np


class Calculation:
    @staticmethod
    def circular_distance(ang1, ang2):
        """
        Compute the circular distance between two angles.
        """

        circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

        return circdist

    @staticmethod
    def proj_a_onto_b(a: np.array, b: np.array):
        """
        Projects vector a onto vector b.
        """
        return b * (np.dot(a, b) / np.linalg.norm(b) ** 2)
