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



##### References
Icon made by Smashicons from www.flaticon.com
