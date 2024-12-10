import numpy as np
import pandas as pd
import scipy.signal as signal


class QualityControl:

    def remove_jumps(dataframe, maxJumpLength):
        """
        Remove large jumps in the x/y position of bodyparts, usually resulting from swaps between animals.
        """

        # get all column names
        scorer = dataframe.columns.get_level_values(0)[0]
        bps = list(
            dataframe.columns.get_level_values(1).unique()
        )  # list(dataframe.columns.levels[1])
        params = list(dataframe.columns.levels[2])
        dataframeMod = dataframe.copy()

        for i, partName in enumerate(bps):

            xDiff = pd.Series(np.diff(dataframe[scorer][partName]["x"]))
            yDiff = pd.Series(np.diff(dataframe[scorer][partName]["y"]))

            xJumpsPositive = signal.find_peaks(xDiff.interpolate(), threshold=200)
            xJumpsNegative = signal.find_peaks(xDiff.interpolate() * -1, threshold=200)
            yJumpsPositive = signal.find_peaks(yDiff.interpolate(), threshold=200)
            yJumpsNegative = signal.find_peaks(yDiff.interpolate() * -1, threshold=200)

            toKill = np.zeros((len(yDiff),), dtype=bool)

            for j in range(len(xJumpsPositive[0])):
                if np.any(
                    (xJumpsNegative[0] > xJumpsPositive[0][j])
                    & (xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength)
                ):
                    endIdx = np.where(
                        (xJumpsNegative[0] > xJumpsPositive[0][j])
                        & (xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength)
                    )
                    toKill[xJumpsPositive[0][j] : xJumpsNegative[0][endIdx[0][0]]] = (
                        True
                    )
                else:
                    toKill[xJumpsPositive[0][j]] = True

            for j in range(len(xJumpsNegative[0])):

                if np.any(
                    (xJumpsPositive[0] > xJumpsNegative[0][j])
                    & (xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength)
                ):
                    endIdx = np.where(
                        (xJumpsPositive[0] > xJumpsNegative[0][j])
                        & (xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength)
                    )
                    toKill[xJumpsNegative[0][j] : xJumpsPositive[0][endIdx[0][0]]] = (
                        True
                    )
                else:
                    toKill[xJumpsNegative[0][j]] = True

            for j in range(len(yJumpsPositive[0])):
                if np.any(
                    (yJumpsNegative[0] > yJumpsPositive[0][j])
                    & (yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength)
                ):
                    endIdx = np.where(
                        (yJumpsNegative[0] > yJumpsPositive[0][j])
                        & (yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength)
                    )
                    toKill[yJumpsPositive[0][j] : yJumpsNegative[0][endIdx[0][0]]] = (
                        True
                    )
                else:
                    toKill[yJumpsPositive[0][j]] = True

            for j in range(len(yJumpsNegative[0])):
                if np.any(
                    (yJumpsPositive[0] > yJumpsNegative[0][j])
                    & (yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength)
                ):
                    endIdx = np.where(
                        (yJumpsPositive[0] > yJumpsNegative[0][j])
                        & (yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength)
                    )
                    toKill[yJumpsNegative[0][j] : yJumpsPositive[0][endIdx[0][0]]] = (
                        True
                    )
                else:
                    toKill[yJumpsNegative[0][j]] = True

            toKill = np.insert(toKill, 1, False)

            dataframeMod.loc[toKill, (scorer, partName, params)] = np.nan

        return dataframeMod
