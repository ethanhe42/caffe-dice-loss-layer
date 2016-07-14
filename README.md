# ultrasound nerve segmentation
Solutions for Kaggle Ultrasound nerve segmentation.  
run `solve_.py` for different models.  
`cfgs_.py` are configurations for different models.  
`model/` folders are `.prototxt` files  
`perClassLoss.py` implement dice loss layer  
`preprocessing.py` implement elastic distortion and random flip  

### Loss
- use dice coeff directly?
- use softmax?
- use euclidean dist(saliency)

### main streams
- SegNet
- saliency detection
- deeplab
- msra
- deep FAIR structure

### related open source codes
https://github.com/fyu/dilation  
https://github.com/fvisin/reseg  
https://github.com/imatge-upc/saliency-2016-cvpr  
https://github.com/yuhuayc/alignhier  
https://github.com/ivpshu/Co-Saliency-Detection-Based-on-Hierarchical-Segmentation  
https://github.com/HyeonwooNoh/DeconvNet  
deeplab  


