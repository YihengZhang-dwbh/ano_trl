# ANO: A Unified RL Framework for Robust Policy Optimization
This repository provides the official implementation of **ANO**,  **Anchored Neighborhood Optimization**.

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
cd ANO/

# Create and activate Conda environment
conda env create -f ano_trl.yml
conda activate ano_trl

````

---

## 🚀 Running Experiments

Use the following command to launch training:

```bash
bash bash_ano.sh
```

## 📊 Eval

Use the following command to launch evaluation:

```bash
bash judge.sh
```

---

## 📎 Cite As

This work has been submitted to ICML 2026. If you use our work, please cite as:

>Anonymous. ANO: A Unified RL Framework for Robust Policy Optimization. In The Forty-Third International Conference on Machine Learning (ICML 2026).


### 📚 BibTeX

```bibtex
@inproceedings{Anonymous2026anon,
  title     = {ANO: A Unified RL Framework for Robust Policy Optimization},
  author    = {Anonymous authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

This repository builds upon and uses code from the following excellent open-source projects. We sincerely thank their authors:

* [TRL](https://github.com/huggingface/trl) – TRL - Transformer Reinforcement Learning.

Please refer to their licenses and cite them if you build upon their work.

---

