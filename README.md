# VectorPD: Artistic Portrait Drawing with Vector Strokes

[![arXiv](https://img.shields.io/badge/arXiv-2410.04182-b31b1b.svg)](https://arxiv.org/abs/2410.04182)

## Introduction
![](repo_images/style.jpg?raw=true)
Editing the brush style on SVGs. Our method generates portrait sketches in vector form, which can be easily used by designers for further editing.

<br>
We define a portrait sketch as a set of Bézier curves and use a differentiable rasterizer ([diffvg](https://github.com/BachiLi/diffvg)) to optimize the parameters of the curves directly with respect to a CLIP-based loss, a Vgg-based loss, and a Crop-based loss. 

Here is a simple visual illustration of our framework.
(More details about loss can be found in the paper)
![](repo_images/details.jpg?raw=true)
<br>

<br>
To generate a clean portrait sketch, we extract the outline of face based on the code from ([here](https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file))

![](repo_images/masked.png?raw=true)

![](repo_images/edge.png?raw=true)

<br>


## Installation
### Installation via pip
```bash
git clone https://github.com/yael-vinker/CLIPasso.git
cd CLIPasso

git clone https://github.com/zllrunning/face-parsing.PyTorch.git
```
```bash
python3.7 -m venv clipsketch
source clipsketch/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
```
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```

```bash
python run_sketching.py --target_file <target_image_name> --mask_file <masked_image_name> --num_strokes 40 --num_sketches 1
```
The resulting sketches will be saved to the "output_sketches" folder, in SVG format.


## Related Work & Code
[CLIPasso](https://arxiv.org/abs/2202.05822): Semantically-Aware Object Sketching, 2022 (Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, Ariel Shamir)

## Citation
If you make use of our work, please cite our paper:

```
@misc{liang2024artisticportraitdrawingvector,
      title={Artistic Portrait Drawing with Vector Strokes}, 
      author={Yiqi Liang and Ying Liu and Dandan Long and Ruihui Li},
      year={2024},
      eprint={2410.04182},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.04182}, 
}
```
