![cabbage](https://user-images.githubusercontent.com/831215/32352134-3af56ab4-c020-11e7-8a6f-4476a25c6626.png)
# cabbage

Unofficial implementation of the paper[1]: [Multiple People Tracking by Lifted Multicut and Person Re-identification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tang_Multiple_People_Tracking_CVPR_2017_paper.pdf)

## Overview
* [Install](https://github.com/justayak/cabbage#install): tools needed for running the application
* [Execute Code](https://github.com/justayak/cabbage#execute-code): how to run the code
  * [DeepMatching](https://github.com/justayak/cabbage#generate-deep-matches): how to generate DeepMatching for your video

![mot16_11](https://user-images.githubusercontent.com/831215/35177494-17453dea-fd80-11e7-92b4-859dde2d6e71.png)
*Tracking calculated by this library on the MOT16-11 video using dmax=100*

## Install

The software is developed using Ubuntu 16.04 and OSX with Python 3.5.
The following libraries and tools are needed for this software to work correctly:

* **tensorflow** (1.x+)
* **Keras** (2.x+)

### Download source tree
Download the source code and its submodules using
```bash
git clone --recursive https://github.com/justayak/cabbage.git
```

### Install-Script
When the above criterias are met a simple install routine can be called inside the
source root
```bash 
bash install.sh
```
This script will create a text file called **settings.txt**. You will need this file when you are
using the end-to-end algorithm.

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

[1] Tang, Siyu, et al. "Multiple people tracking by lifted multicut and person re-identification." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
