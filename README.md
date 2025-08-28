# 💡 SVD-Guided Diffusion for Training-Free Low-Light Image Enhancement

This repository provides the official PyTorch implementation of the paper:  
**["SVD-Guided Diffusion for Training-Free Low-Light Image Enhancement"](https://ieeexplore.ieee.org/document/11122300?source=authoralert)**

Gun Ryu and [Wonjun Kim (Corresponding Author)](https://sites.google.com/view/dcvl)
by **Jingi Kim** and **Wonjun Kim (Corresponding Author)**  

**IEEE Signal Processing Letters**, vol. 32, pp. 3245–3249, 2025.

---

## 📦 Installation

### 🛠 Environment Setup

Install the required dependencies via conda:

```bash
conda env create -f environment.yml
conda activate SGD-release
```

---

## 🚀 Run Inference

To run the model on your dataset:

```bash
python main.py --config llve.yml --path_y ./dataset -i ./result
```

**Arguments:**
- `--config`: path to configuration file (e.g., `llve.yml`)
- `--path_y`: path to the low-light image folder
- `-i`: output directory for enhanced results

---

## 📊 Results

### ✨ Qualitative Results

![Qualitative Results](figures/Fig.svg)

---

## 📎 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{kim2025svd,
  title={SVD-Guided Diffusion for Training-Free Low-Light Image Enhancement},
  author={Kim, Jingi and Kim, Wonjun},
  journal={IEEE Signal Processing Letters},
  year={2025},
  volume={32},
  pages={3245-3249},
  doi={10.1109/LSP.2025.3597558}
}
```

---

## 📫 Contact

If you have any questions or issues, feel free to reach out:

- **Jingi Kim**: [jingi0614@konkuk.ac.kr]  
