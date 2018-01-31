import numpy as np


class VideoData:
    """ representation of a video
    """

    def __init__(self, data):
        """
            data: {np.array} [ (frame, ....), ... ]
        """
        n, m = data.shape
        self.n_detections = n
        self.dimension = m

        self.last_frame = -1
        self.is_ordered = True
        for i in range(1, n):
            frame1 = data[i-1][0]
            frame2 = data[i][0]
            if frame2 < frame1:
                self.is_ordered = False
            if frame2 > self.last_frame:
                self.last_frame = int(frame2)

        self.data = data


    def get_n_first_frames(self, n):
        """
            n: {int} get frames from 1 ... n
        """
        #assert self.is_ordered, 'This code currently only works ordered'
        result = []
        for i in range(0, self.n_detections):
            frame = self.data[i][0]
            if frame > n:
                if self.is_ordered:
                    break
                else:
                    continue
            result.append(self.data[i])


        return np.array(result)
