<img src="images/ezgif-5-3a9525dd52.gif" height="402pt">

# About
Implementation of


* [Learning Geometric Phase Field representations](https://drive.google.com/drive/u/0/folders/1LKQha7mYWvPzKKS2yC0zf_19FEzRlly8) (Yannick Kees 2022)


| File | Description |
| --- | --- |
| `3Dvisualization.ipynb` | Coarse rendering of Neural Networks in Jupyter Notebook |
| `dataset.py` | Creates and Visualises Datasets with the shapes from the shapemaker file |
| `error_decomposition.py` | Plot of different contributions of Loss functional  |
| `different_networksizes.py` | Measure accuracy of NN while increasing networks  |
| `learn_shape_space_ellipse.py` | Training shape space network for ellipsoids |
| `learn_shapespace.py` | Training shape space network for Metaballs  |
| `loss_functionals.py` | Computes Modica-Mortola and Ambrosio-Tortorelli  |
| `misc.py` | Handles import of different file formates, enables CUDA and shows progress on console  |
| `networks.py` | Neural Networks  |
| `packages.py` | All used third party packages |
| `pointclouds.py` | Creates or changes point clouds |
| `run.py` | Solves the 2D reconstruction problem. Can be executed on any computer |
| `Shapemaker.py` | Programm that can produce random point clouds in 2D or 3D form natural looking objects |
| `test_autoencoder.py` | Plot inputs and outputs of Autoencoder for differnt shapes of dataset  |
| `test_shape_space.py` | Make plots of elements of shape space after training  |
| `test.py` | Ignore this.. |
| `train_autoencoder.py` | Train PointNet - Autoencoder for the different datasets  |
| `visualizing.py` | Handles visualization of input and output data |
| `volta.py` | Solves the 3D reconstruction problem. Should only be executed on high performance computer |



# How to install:
1. ssh .... & enter password
2. install conda using wget URL, bash~/Anaconda, conda env list
Then type 
```shell
source ~/anaconda3/bin/activate
conda create -n pytorch3d python=3.10
conda activate pytorch3d
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install matplotlib
pip install random-fourier-features-pytorch 
pip install k3d
git clone https://github.com/paulo-herrera/PyEVTK
cd PyEVTK
python setup.py install
git clone https://github.com/Yannick-Kees/Masterarbeit
cd Masterarbeit
```


Get files from external Computer using 
```
scp IP_ADRESS:~\Masterarbeit\structured2560.vts C:\Users\Yannick\Desktop
```

### External packages:
* [Random Fourier Features Pytorch](https://github.com/jmclong/random-fourier-features-pytorch)  
* [K3D Jupyter](https://github.com/K3D-tools/K3D-jupyter)  
* [EVTK (Export VTK) ](https://github.com/paulo-herrera/PyEVTK) 
* [PointNet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder/tree/cc270113da3f429cebdbe806aa665c1a47ccf0c1) 
* [Park]
* [PyTorch3D](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html)


# Literatur:
Quellen und so
 
## Classical surface reconstruction problem

* Grids: [Poisson Surface Reconstruction](https://hhoppe.com/poissonrecon.pdf) (Michael Kazhdan, Matthew Bolitho and Hugues Hoppe 2006)
* Grids: [Screened poisson surface reconstruction](https://dl.acm.org/doi/10.1145/2487228.2487237) (Michael Kazhdan Hugues Hoppe 2013) <- Video
* Grids: [Fast Surface Reconstruction Using the Level Set Method](https://www.cs.jhu.edu/~misha/Fall05/Papers/zhao01.pdf) (Zhao, Osher, Fedkiw 2013)
* RBF: [Reconstruction and Representation of 3D Objects with Radial Basis
Functions](https://www.cs.jhu.edu/~misha/Fall05/Papers/carr01.pdf) (Carr, Beatson, Cherrie, ... 2001)
* Polynomials: [Multi-level Partition of Unity Implicits](https://www.cc.gatech.edu/~turk/my_papers/mpu_implicits.pdf) (Ohtake, Belyaev, .. 2003)
* Polynomials: [Smoothing of partition of unity implicit surfaces for noise robust surface reconstruction](http://www.den.t.u-tokyo.ac.jp/nagai/Material/PoissonPU/PoissonPU.pdf) (Nagai et all 2009)


 ## Surface reconstruction via occupancy function:

* [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
* [IM-Net: Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1812.02822) (Chen et al. 2018)
* [SAL: Sign Agnostic Learning of Shapes from Raw Data](https://arxiv.org/pdf/1911.10414.pdf) (Atzmon, Lipman 2020)
* [NEURAL UNSIGNED DISTANCE FIELDS FOR IMPLICIT FUNCTION LEARNING](https://virtualhumans.mpi-inf.mpg.de/ndf/) (Chibane, 2020)
* [Convolutional Occupancy Networks](https://www.is.mpg.de/publications/peng2020eccv) (Niemeyer et al. 2018)
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf) (Mildenhall et al. 2020) 


 ## Surface reconstruction via SDF:

* [Phase Transitions, Distance Functions, and Implicit Neural Representations](https://arxiv.org/pdf/2106.07689.pdf) (Yaron Lipman 2021)
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019) 
* [MetaSDF: Meta-Learning Signed Distance Functions](https://www.vincentsitzmann.com/metasdf/) (Sitzman et al. 2019) 
* [Implicit Geometric Regularization for Learning Shapes](https://arxiv.org/pdf/2002.10099.pdf) (Gropp, Lipman et al. 2019) 
* [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/pdf/2003.10983.pdf) (Chabra et al. 2019) 
* [Curriculum DeepSDF](https://arxiv.org/pdf/2003.08593.pdf) (Duan et al. 2019) 
* [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance](https://arxiv.org/pdf/2003.09852.pdf) (Yariv, Lipman et al. 2019) 
* [ON THE EFFECTIVENESS OF WEIGHT-ENCODED NEURAL IMPLICIT 3D SHAPES](https://arxiv.org/pdf/2009.09808v3.pdf) (Davies, Nowrouzezahrai, Jacobson 2021) 


## Application

* [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](https://arxiv.org/pdf/1905.05172.pdf) (Saito et al. 2019)
* [Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes](https://arxiv.org/pdf/2101.10994.pdf) (Takikawa et al. 2021)
* [PhySG: Inverse Rendering with Spherical Gaussians for Physics-based Material Editing and Relighting](https://kai-46.github.io/PhySG-website/) (Zhang et al. 2021)
* [RGB-D Local Implicit Function for Depth Completion of Transparent Objects](https://arxiv.org/pdf/2104.00622.pdf) (Zhu et al. 2021)

## PvJ
* [MARCHING CUBES IN AN UNSIGNED DISTANCE FIELD FOR SURFACE RECONSTRUCTION FROM UNORGANIZED POINT SETS](https://repository.eafit.edu.co/bitstream/handle/10784/9702/2010_Congote_MARCHING_CUBES_UNSIGNED_DISTANCE_FIELD.pdf) (Congote 2010)
* [Shape-Aware Matching of Implicit Surfaces Based on Thin Shell Energies](https://link.springer.com/content/pdf/10.1007/s10208-017-9357-9.pdf) (Iglesias, Rumpf 2015)
* [A Thin Shell Approach to the Registration of Implicit Surfaces](https://ins.uni-bonn.de/media/public/publication-media/IgBeRu13.pdf) (Iglesias, Rumpf 2013)
* [Geometry Processing with Neural Fields](http://vladlen.info/papers/neural-fields.pdf) (Yang 2021)

## Other important papers
* [A constructive geometry for computer graphics](https://watermark.silverchair.com/160157.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsQwggLABgkqhkiG9w0BBwagggKxMIICrQIBADCCAqYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMZ3yC8S9z4j16q6adAgEQgIICd93WBkm4nz7RugxL5LvwZ3R_Hk_PR0Q333c_JNhOyjY0tEOwaVnK4H3JVvik5EWE0NfD6KX21Db0G9VhHjfdEeTEHITAbNip607L5K7TxDSmM5MZV2aBM-AHH04Psu6Qtkl98sqt3NfnO1f6jLyWqVx49oacCQ0tOZ4rAajU4w7CWJjQ95zP_qSmsaf74mgNYk6bYInYkvJOaU1LtVtSNkkdodu5a7q2NdEATvngabIQfsN03i7iPLiRCg1Oh1cfEGkrBScEO_cycYRTq4cKPfaqFcJKoLxtOaYnd_HPETC_LtfhncVQcVXItuSFlDLfrTXweitXeoSC5-lgjCbnM1jyYtFgrucVCADW_fYcHcFTHEhJIiDHLGAbt4nreSSHXtCS5AHA2GpMB1WgoEf00__hHJdF8GXgD6p1rIN0DjxPHsf2EkF_kg_2DaPdfm4T6XvX3-Bz_vdn4z3dv9AYyYT6zd46RBXyjUaaCFphFs8mpNMYCuJPieS_YF9bpMjp_3qUzIWEFSMc2_coe770WCws89aphGf2tCAfhsPJFDxxdo4bl1pYk6Lc9PN3dIAx3V5vlqHMIKfAB-BxjNsA7aLY024Ar41qJ14WfHgZ-Emxzpe88kf42E-x4qJ7BP6x_nr7pn2wyIa6nKKgl3clNZEy-6LPmhvVb3ccPHx0-cbY6SAv3AQpQxHbESNu-7jJ0b8pGXGdv3DfWYbLZnZCzcZhUlgOwKyCUqQrwcQJ_s-PGdxvMtpanXVKLIOgHgXN1jwmWt-WCOYpCkFj_TJTRoN7qDtm_Xfplf_EqqLGA1pZryo1gUYSu50fKfCnYB7btRuwIN4yPIM) (Ricci 1974)
* [Representation and Rendering of Implicit Surfaces ](http://www.13thmonkey.org/documentation/CAD/Sig06.pdf) (Sigg 2006 PhD)
* [Metaballs](https://xbdev.net/misc_demos/demos/marching_cubes/paper.pdf) (Kenwright T.A. 2014)
* [Spelunking the Deep: Guaranteed Queries on General Neural Implicit Surfaces via Range Analysis](https://arxiv.org/pdf/2202.02444.pdf) (Sharp 2022)
* [DeepCurrents: Learning Implicit Representations of Shapes with Boundaries ](https://people.csail.mit.edu/smirnov/deep-currents/) (Palmer 2022)
* [SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization](https://igl.ethz.ch/projects/SAPE/SAPE_paper.pdf) (Hertz 2021)
* [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://arxiv.org/pdf/2012.02189.pdf) (Tancik 2021)


## MtbA

* [Spelunking the Deep: Guaranteed Queries on General Neural Implicit Surfaces via Range Analysis](https://arxiv.org/pdf/2202.02444.pdf) (SHARP 2022)
* [MIP-plicits: Level of Detail Factorization of Neural Implicits Sphere Tracing](https://arxiv.org/pdf/2201.09147.pdf) (Silva 2022)
* [Seeing Implicit Neural Representations as Fourier Series](https://arxiv.org/pdf/2109.00249.pdf) (Benbarka 2021)
* [COIN: COmpression with Implicit Neural representations](https://neuralfields.cs.brown.edu/paper_120.html) (Emilien Dupont 2021)
* [DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction](https://arxiv.org/pdf/1905.10711.pdf) (Wang 2021)

