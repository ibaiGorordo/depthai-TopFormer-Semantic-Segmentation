# depthai TopFormer Semantic Segmentation
 Python scripts performing on devive semantic segmentation (ADE20K 150 classes)using the TopFormer model in depthai.


![!TopFormer Semantic Segmentation](https://github.com/ibaiGorordo/depthai-TopFormer-Semantic-Segmentation/blob/main/doc/img/output.png)
*Taken from: https://youtu.be/fzZEylhZTbI*

# Requirements

 * Check the **requirements.txt** file. 
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/depthai-TopFormer-Semantic-Segmentation.git
cd depthai-TopFormer-Semantic-Segmentation
pip install -r requirements.txt
```

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# MyriadX models
The original Pytorch models were converted to MyriadX blob using the Google Colab notebook below. You can download the converted models from [Google Drive](https://drive.google.com/drive/folders/1is_eQOVYd_bLP4vO4uAHdCj7byRlzBu7?usp=sharing) and save them into the [modes folder](https://github.com/ibaiGorordo/depthai-TopFormer-Semantic-Segmentation/tree/main/models).
- **Conversion notebook**: TopFormer_to_MyriadX_Conversion.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UuPGpIpFLC2aN2gpsBLkgOcm6WdShIaE?usp=sharing) 
- The License of the models is Apache-2.0 License: https://github.com/hustvl/TopFormer/blob/main/LICENSE

# ONNX Inference
The model can also run in ONNX using the scripts in [this repository](https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation).

# Pytorch model
The original Pytorch model can be found in this repository: https://github.com/hustvl/TopFormer
 
# Examples

 * **Depthai Semantic Segmentation** using the camera in the board (~20 fps for small model):
 ```
 python main.py
 ```

 * **Video inference**: https://youtu.be/x3UVNlPlFlc
 ```
 python main_video.py
 ```
 ![!CREStereo depth estimation Depthai](https://github.com/ibaiGorordo/depthai-TopFormer-Semantic-Segmentation/blob/main/doc/img/topformer_depthai.gif)
  
 *Original video: https://youtu.be/fzZEylhZTbI*

# References:
* TopFormer model: https://github.com/hustvl/TopFormer
* ONNX TopForfmer: https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation
* Luxonis Depthai: https://docs.luxonis.com/en/latest/
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* Original paper: https://arxiv.org/abs/2204.05525
