import numpy as np
from time import time
from os import makedirs, listdir, remove
from os.path import join, isfile, isdir, exists, splitext
from cabbage.features.GenerateFeatureVector import pairwise_features
from cabbage.data.video import VideoData
from cabbage.features.ReId import get_element
from time import time
from keras.applications.vgg16 import preprocess_input
import cabbage.features.spatio as st


class AABBLookup:
    """ helper function to easier map the AABB's
    """

    def __init__(self, Dt, X, H=64, W=64):
        """ctor
            Dt: {np.array} detections for the video
                -> [(frame, x, y, w, h, score), ...]

            X: {np.array} (n, w, h, 3) video of the detections
        """
        n, m = Dt.shape
        assert m == 6

        self.Im = np.zeros((n, H, W, 3), 'uint8')
        self.AABBs = np.zeros((n, 4), 'float32')
        IDS_IN_FRAME = [None] * (n + 1)  # frames start at 1 and not at 0
        self.Scores = [0] * n
        self.Frames = np.array([0] * n)

        for i, (frame, x, y, w, h, score) in enumerate(Dt):
            frame = int(frame)
            im = get_element(X[frame-1], (x,y,w,h), (W, H), True)
            self.Im[i] = im
            self.AABBs[i] = np.array([x,y,w,h])
            self.Scores[i] = score
            self.Frames[i] = frame

            if IDS_IN_FRAME[frame] is None:
                IDS_IN_FRAME[frame] = []
            IDS_IN_FRAME[frame].append(i)

        self.ids_in_frame = IDS_IN_FRAME
        self.Scores = np.array(self.Scores)

    def __getitem__(self, i):
        return self.AABBs[i], self.Im[i], self.Scores[i], self.Frames[i]




class BatchGraphGenerator:
    """
    """

    def __init__(self, root, reid, dm, dmax, video_name):
        """
        """
        self.data_loc = join(root, 'BATCH_GG_' + video_name + "_dmax_" + str(dmax))
        if not isdir(self.data_loc):
            makedirs(self.data_loc)
        self.reid = reid
        self.dmax = dmax
        self.video_name = video_name
        self.dm = dm


    def build(self, Dt, X, W, batch_size=500):
        """ build the graph
            Dt: {np.array} detections for the video
                -> [(frame, x, y, w, h, score), ...]

            X: {np.array} (n, w, h, 3) video of the detections
        """
        dmax = self.dmax
        lifted_edge_start = int(dmax/2)

        __start = time()
        lookup = AABBLookup(Dt, X)
        __end = time()
        print('create lookup structure, elapsed:', __end - __start)

        n, _ = Dt.shape

        ALL_PAIRS = []
        LAST_FRAME = X.shape[0]

        __start = time()
        for frame_i, ids_in_frame in enumerate(lookup.ids_in_frame):
            if ids_in_frame is None:
                continue

            for i in ids_in_frame:
                for j in ids_in_frame:
                    if i < j:
                        ALL_PAIRS.append((i,j))

                for frame_j in range(frame_i + 1, min(frame_i + dmax + 1, LAST_FRAME)):
                    if lookup.ids_in_frame[frame_j] is None:
                        continue
                    for j in lookup.ids_in_frame[frame_j]:
                        if j > i:
                            ALL_PAIRS.append((i, j))

            if frame_i % 100 == 0:
                print('handle frame ' + str(frame_i) + " from " + str(LAST_FRAME))

        __end = time()
        ALL_PAIRS = np.array(ALL_PAIRS, 'int32')
        print("ALL PAIRS:", ALL_PAIRS.shape)
        print('\telapsed seconds:', __end - __start)

        reid = self.reid
        dm = self.dm
        video_name = self.video_name


        edge_file, lifted_edge_file = self.get_file_names()
        EDGE_FILE = open(edge_file, "w")
        LIFTED_EDGE_FILE = open(lifted_edge_file, "w")


        for _i in range(0, len(ALL_PAIRS), batch_size):
            __start = time()
            batch = ALL_PAIRS[_i:_i+batch_size]
            i,j = batch[:,0],batch[:,1]
            aabb_j, Im_j, scores_j, frame_j = lookup[j]
            aabb_i, Im_i, scores_i, frame_i = lookup[i]

            delta = frame_j - frame_i
            IN_RANGE = (delta < dmax).nonzero()
            delta = delta[IN_RANGE]

            aabb_j, aabb_i = aabb_j[IN_RANGE], aabb_i[IN_RANGE]
            scores_i, scores_j = scores_i[IN_RANGE], scores_j[IN_RANGE]
            frame_i, framej = frame_i[IN_RANGE], frame_j[IN_RANGE]

            Im_j, Im_i = \
                preprocess_input(Im_j[IN_RANGE].astype('float64')), \
                preprocess_input(Im_i[IN_RANGE].astype('float64'))

            SCORES = np.where(scores_j < scores_i, scores_j, scores_i)

            ST = np.array(
                [st.calculate(bb1, bb2) for bb1, bb2 in zip(aabb_i, aabb_j)]
            )

            DM = np.array(
                [dm.calculate_cost(video_name, f1, bb1, f2, bb2) for \
                    f1, bb1, f2, bb2 in zip(frame_i, aabb_i, frame_j, aabb_j)]
            )

            Y = reid.predict_raw(np.concatenate([Im_i, Im_j], axis=3))[:,0]

            Bias = np.ones(ST.shape)

            # delta = delta[IN_RANGE]
            # Bias = Bias[IN_RANGE]
            # ST = ST[IN_RANGE]
            # DM = DM[IN_RANGE]
            # Y = Y[IN_RANGE]
            # SCORES = SCORES[IN_RANGE]

            assert np.min(delta) >= 0

            F = np.array([
                Bias,
                ST, DM, Y, SCORES,
                ST**2, ST * DM, ST * Y, ST * SCORES,
                DM**2, DM * Y, DM * SCORES,
                Y**2, Y * SCORES,
                SCORES**2
            ]).T

            edge_weights = np.einsum('ij,ij->i', F, W[delta])

            for i, j, w, d in zip(i, j, edge_weights, delta):
                txt = str(i) + " " + str(j) + " " + str(w) + "\n"
                if d < lifted_edge_start:
                    EDGE_FILE.write(txt)
                    EDGE_FILE.flush()
                else:
                    LIFTED_EDGE_FILE.write(txt)
                    LIFTED_EDGE_FILE.flush()

            __end = time()
            print('finish batch ' + str(_i) + ' .. ' + str(_i+batch_size) + \
                " total:" + str(len(ALL_PAIRS)) + " ... elapsed:" + \
                    str(__end - __start))


        EDGE_FILE.close()
        LIFTED_EDGE_FILE.close()

    def get_file_names(self):
        """
        """
        edge_file = join(self.data_loc, "edges.txt")
        lifted_edge_file = join(self.data_loc, "lifted_edges.txt")
        return edge_file, lifted_edge_file




class GraphGenerator:
    """
    """


    def __init__(self, root, video, detections, dmax, W, d=None,
        video_name=None, DM_object=None, reid_object=None,
        is_memorized_reid=False):
        """
            root: {string} path to data root
            detections: {np.array} ([f, x, y, w, h, score], ...)
            dmax: {int}
            W: {np.array} calculated theta scores

        """
        if d is None:
            d = int(dmax/2)
        assert dmax > d
        #assert W.shape[0] == dmax

        if video_name is None:
            video_name = "video" + str(time)

        self.root = root
        self.X = video
        self.dmax = dmax
        self.d = d
        self.detections = detections
        self.video_name = video_name

        data_loc = self.get_data_folder()
        if not isdir(data_loc):
            makedirs(data_loc)

        n, _ = self.detections.shape
        #edges = []
        #lifted_edges = []

        start_i, edges, lifted_edges = self.load_edges(data_loc)

        ALL_EDGES = []

        gen = pairwise_features(self.root,None,
            DM_object=DM_object, reid_object=reid_object)

        vd = VideoData(detections)
        is_ordered = vd.is_ordered
        #is_ordered = self.check_if_detections_are_ordered(detections)

        #for i, entry in enumerate(self.detections):
        for i in range(start_i+1, n):
            __START = time()
            frame1, x1, y1, w1, h1,conf1 = detections[i]
            I1 = self.X[int(frame1-1)]
            for j in range(i+1, n):
                frame2, x2, y2, w2, h2,conf2 = detections[j]
                delta = int(abs(frame2-frame1) )
                if delta >= dmax :
                    if is_ordered:
                        break
                    else:
                        continue

                I2 = self.X[int(frame2-1)]

                try:
                    i1 = i if is_memorized_reid else None
                    i2 = j if is_memorized_reid else None
                    vec = gen.get_pairwise_vector(
                            video_name ,
                            I1, I2,
                            frame1,frame2,
                            (x1, y1, w1, h1),
                            (x2, y2, w2, h2),
                            conf1,
                            conf2,
                            i1=i1, i2=i2)

                    cost = -1 * (W[delta]@np.array(vec))

                    if delta > d:
                        if (cost > 0):
                            # lifted edge
                            lifted_edges.append((i,j,cost))
                    else:
                        # normal edge
                        edges.append((i,j,cost))

                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    print("ignore frame " + str(frame1) + " -> " + str(frame2) \
                        +' because ' + str(e) + ' ... delta:' + str(delta))
                #print('cost:', cost)
                #cost = 10 if pid == pid_o else -1
            __END = time()
            print("edges for detection: ",i," out of ",n)
            print("\t\telapsed:", __END - __START)

            # --- store the data ---
            self.save_edges(i, data_loc, edges, lifted_edges)

            edge_OLD, lifted_edges_OLD = self.get_backup_file_names(i-1, data_loc)
            if isfile(edge_OLD):
                remove(edge_OLD)
            if isfile(lifted_edges_OLD):
                remove(lifted_edges_OLD)


        edges = np.array(edges)
        lifted_edges = np.array(lifted_edges)

        print('Edges', edges.shape)
        print('Lifted Edges', lifted_edges.shape)


        with open('config.txt', 'w+') as f:
            print(str(n), file=f)

        fmt = '%d %d %f'
        np.savetxt('edges.txt', edges, delimiter=';', fmt=fmt)
        np.savetxt('lifted_edges.txt', lifted_edges, delimiter=';', fmt=fmt)


    def get_backup_file_names(self, i, backup_loc):
        """
        """
        edge_file = join(backup_loc, "edges_%06d" % (i,) + '.npy')
        lifted_edge_file = join(backup_loc, "lifted_edges_%06d" % (i,) + '.npy')
        return edge_file, lifted_edge_file


    def get_i(self, backup_loc):
        edges = sorted([f for f in listdir(backup_loc) if
            f.startswith('edges') and f.endswith('.npy')])
        if len(edges) > 0:
            last_edge = edges[-1]
            return int(last_edge[6:12])
        else:
            return -1


    def load_edges(self, backup_loc):
        """
        """
        i = self.get_i(backup_loc)
        if i > 0:
            edge_file, lifted_edge_file = self.get_backup_file_names(i, backup_loc)
            assert isfile(edge_file)
            edges = np.load(edge_file).tolist()

            if isfile(lifted_edge_file):
                lifted_edges = np.load(lifted_edge_file).tolist()

        else:
            edges = []
            lifted_edges = []

        return i, edges, lifted_edges


    def save_edges(self, i, backup_loc, edges, lifted_edges):
        """
        """
        edge_file, lifted_edge_file = self.get_backup_file_names(i, backup_loc)
        assert len(edges) > 0
        np.save(edge_file, edges)

        if len(lifted_edges) > 0:
            np.save(lifted_edge_file, lifted_edges)


    # def check_if_detections_are_ordered(self, detections):
    #     """ Yields true if the list of detections is ordered
    #     """
    #     n, _ = detections.shape
    #     for i in range(1, n):
    #         frame1, x1, y1, w1, h1,conf1 = detections[i-1]
    #         frame2, x2, y2, w2, h2,conf2 = detections[i]
    #         if frame2 < frame1:
    #             return False
    #     return True

    def get_data_folder(self):
        """ gets the data directory
        """
        return join(join(self.root, 'graph_generator'), self.video_name)
