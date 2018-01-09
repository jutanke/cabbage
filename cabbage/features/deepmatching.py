import numpy as np
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
import subprocess
from math import ceil, floor
from pppr import aabb


class DeepMatching:
    """ deep matching (http://lear.inrialpes.fr/src/deepmatching/)
    """

    def __init__(self, deep_matching_binary, data_loc, delta_max):
        """
        """
        assert isfile(deep_matching_binary), 'the deep matching binary must exist'
        self.deepmatch_loc = deep_matching_binary

        if not isdir(data_loc):
            makedirs(data_loc)
        self.data_loc = data_loc
        self.delta_max = delta_max


    def calculate_cost(self, video_name, frame1, bb1, frame2, bb2):
        """ calculates the costs that are needed for the Lifted Graph Cut

            frame1: {int}
            bb1: {aabb}: (x, y, w, h)
            frame2: {int}
            bb2: {aabb}
        """
        M = self.get_match(video_name, frame1, frame2)
        is_same_frame = frame1 == frame2

        intersections = 0
        unions = 0

        for x,y, u, v, _, _ in M:
            p = (x,y)
            q = (u,v)
            if is_same_frame:
                inside_one = aabb.is_inside(bb1, p)
                inside_two = aabb.is_inside(bb2, p)
            else:
                inside_one = aabb.is_inside(bb1, p)
                inside_two = aabb.is_inside(bb2, q)

            if inside_one and inside_two:
                intersections += 1
            if inside_one or inside_two:
                unions += 1

        if unions > 0:
            f_dm = intersections/unions
        else:
            f_dm = 0.5  # uncertain

        if f_dm <= 0:
            f_dm = 1e-10  # to prevent numerical issues

        return f_dm


    def get_match(self, video_name, frame1, frame2):
        """ gets the deep match between two frames in the given video
        """
        assert ceil(frame1) == floor(frame1)
        assert ceil(frame2) == floor(frame2)
        frame1, frame2 = int(frame1), int(frame2)
        assert frame1 <= frame2, "First frame must be greater/equal"
        M = self.get_matches(video_name, frame1)
        n = len(M)
        m = frame2 - frame1
        assert n > m, str(frame2) + " is out of bound for frame " + str(frame1)
        return M[m]


    def get_matches(self, video_name, frame_nbr):
        """ restores the deep matches between a frame and its d_max successors

            video_name: {string} same as in {generate_matches}
            frame_nbr: {int} frame number
        """
        folder_name = self._get_video_folder_name(video_name)
        fname = self._create_file_name_for_frame(frame_nbr)
        assert isdir(folder_name), 'Data location ' + folder_name + ' does not exist'
        file_name = join(folder_name, fname)
        assert isfile(file_name), 'Frame ' + str(frame_nbr) + ' not in ' + folder_name

        M = np.load(file_name)
        return M


    def generate_matches(self, video_folder, video_name,
        img_type='jpg', verbose=True, force_overwrite=False):
        """ generate matches for each frame F from itself to F+delta_max

            video_folder: {string} folder to the images that are to be processed.
                The file names should be ascending with regards to frame number,
                e.g. 00000001.jpg, 00000002.jpg, ... etc

            video_name: {string} name of the video, needed for geneating the
                folder

            delta_max: {int} max number of frames to calculate the deep matches
                for

            force_overwrite: {boolean} if True than the function will fail
                if their is data already collected. If False the function will
                resume where the collection stopped
        """
        delta_max = self.delta_max
        frames = sorted([join(video_folder, f) for f in listdir(video_folder) \
                          if f.endswith('.'+img_type)])

        folder_name = self._get_video_folder_name(video_name)
        if isdir(folder_name):
            assert not force_overwrite, 'Target directory is not empty!'
            already_calc = [f for f in listdir(folder_name) if f.endswith('.npy')]
            start_i = len(already_calc)
        else:
            makedirs(folder_name)
            start_i = 0

        for i in range(start_i, len(frames)):
            curr_frame = []
            for j in range(i, min(i+delta_max+1, len(frames))):
                if verbose:
                    print("{DM}: solve " + str(i+1) + " -> " + str(j+1))
                M = self.deepmatch(frames[i], frames[j])
                curr_frame.append(M)

            fname = self._create_file_name_for_frame(i+1)
            np.save(join(folder_name, fname), np.array(curr_frame))


    def deepmatch(self, img1, img2):
        """ matches two images

            img1: {string} path to first file
            img2: {string} path to second file
        """
        args = (self.deepmatch_loc, img1, img2, '-downscale', '3', '-nt', '16')
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        B = np.fromstring(popen.stdout.read(), sep=' ')
        n = B.shape[0]
        assert(floor(n) == ceil(n))
        assert(floor(n/6) == ceil(n/6))
        B = B.reshape((int(n/6), 6))
        return B

    # --------------------------------------------------
    # private functions
    # --------------------------------------------------

    def _get_video_folder_name(self, video_name):
        """ private function
        """
        return join(self.data_loc, 'DM_' + video_name + "_dmax" + \
            "%03d" % (self.delta_max,))


    def _create_file_name_for_frame(self, frame):
        """ private function
        """
        return "f" + "%06d" % (frame,) + '.npy'
