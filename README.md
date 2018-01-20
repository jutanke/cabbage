# cabbage
![cabbage](https://user-images.githubusercontent.com/831215/32352134-3af56ab4-c020-11e7-8a6f-4476a25c6626.png)
Implementation of the paper: [Multiple People Tracking by Lifted Multicut and Person Re-identification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tang_Multiple_People_Tracking_CVPR_2017_paper.pdf)

## Install

The following libraries and tools are needed for this software to work correctly:

* **tensorflow** (1.x+)
* **Keras** (2.x+)
* **pak**: (handling of the dataset and evaluation scheme)
```bash
pip install git+https://github.com/justayak/pak.git
```
* **cselect**: (generate n different colors for better visualization)
```bash
pip install git+https://github.com/justayak/cselect.git
```
* **pppr**: (helper functions for geometry)
```bash
pip install git+https://github.com/justayak/pppr.git
```

### Download source tree
Download the source code and its submodules using
```bash
git clone --recursive https://github.com/justayak/cabbage.git
```

## Execute Code
Follow this steps to do an end-to-end run on a video:


### Generate Deep Matches
Download the last DeepMatching Software from [TOTH](http://lear.inrialpes.fr/src/deepmatching/) and unzip it. 
Build the binary using their instructions. **Important**: this operation might run for a long time!

```python
from cabbage.features.deepmatching import DeepMatching

dmax = 100  # as described in the paper
deep_matching_binary = '/path/to/TOTH/deep/matching/binary'
data_loc = '/path/where/data/can/be/stored'

dm = DeepMatching(deep_matching_binary, data_loc, dmax)

# the video must be stored as a folder with images 
video_folder = '/path/to/video/as/img'
video_name = 'Some_Name'
dm.generate_matches(video_folder, video_name)

# query..
# usually, you would not need to do this as it is
# handled within the application
frame1 = 1
bb1 = (10, 10, 10, 50)  # (top-x, top-y, w, h)

frame2 = 3
bb2 = (11, 13, 12, 53)  # (top-x, top-y, w, h)

cost = dm.calculate_cost(video_name, frame1, bb1, frame2, bb2)

```

# References
Icon made by Smashicons from www.flaticon.com
