# DiffusionWorldViewer

This repository contains code for the paper:[DiffusionWorldViewer: Exposing and Broadening the Worldview Reflected by Generative Text-to-Image Models](https://arxiv.org/abs/2309.09944)

<br>
Authors: Zoe De Simone, Angie Boggust, Arvind Satyanarayan, Ashia Wilson

![Teaser](img/Dashboard_UI.jpg)
<br>

Generative text-to-image (TTI) models produce high-quality images from short textual descriptions and are widely used in academic and creative domains. Like humans, TTI models have a worldviews, a conception of the world learned from their training data and task that influences the images they generate for a given prompt. However, the worldviews of TTI models are often hidden from users, making it challenging for users to build intuition about TTI outputs, and they are often misaligned with users' worldviews, resulting in output images that do not match user expectations. In response, we introduce DiffusionWorldViewer, an interactive interface that exposes a TTI model's worldview across output demographics and provides editing tools for aligning output images with user perspectives. In a user study with 18 diverse TTI users, we find that  DiffusionWorldViewer helps users represent their varied viewpoints in generated images and challenge the limited worldview reflected in current TTI models.
<br>

## Demo

A demo of the DiffusionWorldViewer can be run in Google Colab, by running this 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zoedesimone/DiffusionWorldViewer/blob/main/DiffusionWorldViewer_Paper.ipynb)

## Citation
If you find the Embedding Comparator useful in your work, please cite:

```bibtex
@misc{desimone2023fair,
      title={DiffusionWorldViewer: Exposing and Broadening the Worldview Reflected by Generative Text-to-Image Models}, 
      author={Zoe De Simone and Angie Boggust and Arvind Satyanarayan and Ashia Wilson},
      year={2023},
      eprint={2309.09944},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
