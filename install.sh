pip3 install tensorflow scikit-learn keras scikit-image

pip3 install git+https://github.com/justayak/cabbage.git
pip3 install git+https://github.com/justayak/pak.git
pip3 install git+https://github.com/justayak/cselect.git
pip3 install git+https://github.com/justayak/pppr.git

mkdir build
(cd build && cmake .. && make -j8)

if [ ! -d "deepmatching_bin" ]; then
    mkdir deepmatching_bin
    (cd deepmatching_bin && wget http://lear.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip && unzip deepmatching_1.2.2.zip)
fi

python build_settings.py
