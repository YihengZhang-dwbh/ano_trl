# Anon: Exploring the Adaptivity of Optimizers and Beyond
This repository provides the official implementation of **Anon**, an **A**daptivity **N**on-restricted **O**ptimizer with **N**ovel convergence technique called **Incremental Delay Update (IDU)**. Anon bridges the gap between SGD-like and Adam-like behaviors, achieving state-of-the-art results across vision, language, and generative tasks.

---
## 🔍 Why Anon?

Studies show that adaptive optimizers (e.g., Adam) converge faster but often generalize worse than non-adaptive ones (e.g., SGD). This **generalization gap** is closely linked to the *adaptivity* of the optimizer—how it scales gradients based on historical signal.

Anon introduces:

* **Tunable Adaptivity**: By adjusting γ, Anon smoothly interpolates between different optimization behaviors:

  * γ > 1: escape from saddle points
  * γ = 1: Adam-like
  * γ = 0: SGD-like
  * γ < 0: preference for flatter minima

* **Incremental Delay Update (IDU)**: A novel convergence technique that ensures **stability** across all \$\gamma \in \mathbb{R}\$ without relying on hard constraints like AMSGrad’s max operation.

---
## 📈 Visualizations

To better understand how different optimizers behave in complex landscapes, we visualize their trajectories on two classical benchmark functions: **Rosenbrock** and **Rastrigin**. These functions are widely used to evaluate the optimizer's ability to escape saddle points, navigate flat valleys, and avoid local minima. Rosenbrock tests the optimizer's capacity to follow narrow curved paths toward a global minimum, while Rastrigin challenges it with a rugged landscape filled with deceptive local minima.

| Rastrigin Function | Rosenbrock Function |
| :----------------: | :-----------------: |
| <img src="pic/1.png" width="300" height="300"/> | <img src="pic/2.png" width="300" height="300"/> |

- **Rastrigin**: A highly non-convex function with many local minima. The global minimum is at **(0, 0)**.
- **Rosenbrock**: A narrow, curved valley with the global minimum at **(1, 1)**. It's commonly used to evaluate optimizer stability and curvature sensitivity.

---

Below are qualitative comparisons of optimization trajectories using different optimizers over benchmark functions:

### 📌  Rastrigin / Rosenbrock (3D Trajectories)

#### 🔹 Adaptivity Sweep (Anon with Different γ)

|               γ=-0.5               |              γ=0.0              |               γ=0.5               |              γ=1.0              |               γ=1.5               |
| :--------------------------------: | :-----------------------------: | :-------------------------------: | :-----------------------------: | :-------------------------------: |
| <img src="pic/rastrigin3d_-0.5_Anon.png" width="170" height="170"/> | <img src="pic/rastrigin3d_0_Anon.png" width="170" height="170"/> | <img src="pic/rastrigin3d_0.5_Anon.png" width="170" height="170"/> | <img src="pic/rastrigin3d_1_Anon.png" width="170" height="170"/> | <img src="pic/rastrigin3d_1.5_Anon.png" width="170" height="170"/> |

#### 🔹 Optimizer Comparison on Rastrigin

|              AdaBelief             |              AdaBound             |              Adahessian             |              Adam             |              RAdam             |              SGD             |              Yogi             |
| :--------------------------------: | :-------------------------------: | :---------------------------------: | :---------------------------: | :----------------------------: | :--------------------------: | :---------------------------: |
| <img src="pic/rastrigin3d_AdaBelief.png" width="120" height="120"/> | <img src="pic/rastrigin3d_AdaBound.png" width="120" height="120"/> | <img src="pic/rastrigin3d_Adahessian.png" width="120" height="120"/> | <img src="pic/rastrigin3d_Adam.png" width="120" height="120"/> | <img src="pic/rastrigin3d_RAdam.png" width="120" height="120"/> | <img src="pic/rastrigin3d_SGD.png" width="120" height="120"/> | <img src="pic/rastrigin3d_Yogi.png" width="120" height="120"/> |

#### 🔹 Optimizer Comparison on Rosenbrock

|                γ=-0.5               |               γ=0.0              |                γ=0.5               |               γ=1.0              |                γ=1.5               |
| :---------------------------------: | :------------------------------: | :--------------------------------: | :------------------------------: | :--------------------------------: |
| <img src="pic/rosenbrock3d_-0.5_Anon.png" width="170" height="170"/> | <img src="pic/rosenbrock3d_0_Anon.png" width="170" height="170"/> | <img src="pic/rosenbrock3d_0.5_Anon.png" width="170" height="170"/> | <img src="pic/rosenbrock3d_1_Anon.png" width="170" height="170"/> | <img src="pic/rosenbrock3d_1.5_Anon.png" width="170" height="170"/> |

|              AdaBelief              |              AdaBound              |              Adahessian              |              Adam              |              RAdam              |              SGD              |              Yogi              |
| :---------------------------------: | :--------------------------------: | :----------------------------------: | :----------------------------: | :-----------------------------: | :---------------------------: | :----------------------------: |
| <img src="pic/rosenbrock3d_AdaBelief.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_AdaBound.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_Adahessian.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_Adam.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_RAdam.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_SGD.png" width="120" height="120"/> | <img src="pic/rosenbrock3d_Yogi.png" width="120" height="120"/> |

---

## 📋 Environment Setup

**Tested with:**
- **OS:** Ubuntu 20.04  
- **Python:** 3.8+  
- **CUDA:** *(optional, recommended for LLM/diffusion experiments)*

### 🔧 Step-by-step Installation
```bash
# Clone the repository
git clone https://github.com/xxxxxx
cd Anon/ResNet18

# Create and activate Conda environment
conda env create -f environment.yml
conda activate anon_resnet18

````

---

## 🚀 Running Experiments

Use the following command to launch training and evaluation:

```bash
bash run.sh
```


## 📌 Notes

* All hyperparameters **except** `--gamma` are shared with other optimizers.
* Anon’s adaptivity can be tuned via `--gamma`, e.g.:

  * `--gamma 1.1`($\geq 1$): Faster escape from the saddle point
  * `--gamma 1.0`: Adam-like
  * `--gamma 0.0`: SGD-like
  * `--gamma -0.1`($\leq 0$): Flatter minima preference

---

## 📊 Benchmarking

This implementation supports:

* 🧠 **Image Classification**:
  ResNet18/ResNet50 on ImageNet, CIFAR-10

* 🎨 **Image Generation**:
  Diffusion model (DDPM) on CIFAR-10

* 💬 **Language Modeling**:
  GPT2-small / GPT2-medium on OpenWebText

For detailed experimental setups, refer to `xxx.sh` in each subfolders.

---

## 📎 Cite As

This work has been submitted to ICML 2026. If you use our work, please cite as:

>Anonymous. Anon: Exploring the Adaptivity of Optimizers and Beyond. In The Forty-Third International Conference on Machine Learning (ICML 2026).


### 📚 BibTeX

```bibtex
@inproceedings{Anonymous2026anon,
  title     = {Anon: Exploring the Adaptivity of Optimizers and Beyond},
  author    = {Anonymous authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

This repository builds upon and uses code from the following excellent open-source projects. We sincerely thank their authors:

* [AdaHessian](https://github.com/amirgholami/adahessian) – Second-order adaptive optimizer for deep learning.
* [AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer) – Optimizer based on trust in current gradients.
* [Lookaround](https://github.com/Ardcy/Lookaround) – Enhanced curvature-aware optimization.
* [Improved Diffusion](https://github.com/openai/improved-diffusion) – Reference implementation for DDPM models.
* [Sophia](https://github.com/Liuhong99/Sophia) – Fast and scalable optimizer for LLMs.
* [nanoGPT](https://github.com/karpathy/nanoGPT) – Clean GPT training codebase.

Please refer to their licenses and cite them if you build upon their work.

---

