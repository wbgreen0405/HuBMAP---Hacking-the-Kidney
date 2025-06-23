# HuBMAP - Hacking the Kidney

Code for the Kaggle challenge to detect functional tissue units (FTUs) across different tissue preparation pipelines.

[HuBMAP Challenge on Kaggle](https://www.kaggle.com/competitions/hubmap-hacking-the-kidney)

---

## 🏆 Competition Results

- **Diversity Award Winner** 🎉  
  We won the **Diversity Winner** prize!  
  [HuBMAP - Hacking the Kidney Diversity Winner: 404! (YouTube)](https://www.youtube.com/watch?v=pZTjzaP12Sc)
- **Final Placement:** 242 out of 1,200 teams

---

## 🛠️ Project Structure

```
.
├── data/                 # (NOT uploaded) Instructions for getting data
├── img/                  # Visualizations, sample images, etc.
├── notebooks/            # Jupyter notebooks for EDA & model development
│   ├── eda.ipynb
│   ├── baseline_model.ipynb
│   └── ...
├── src/                  # Python source code (data processing, model, training, utils)
│   ├── augmentations.py
│   ├── config.py
│   ├── dataset.py
│   ├── losses.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── logs/                 # Log files
├── outputs/              # Model weights, predictions, submissions
├── tests/                # Unit and integration tests
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_losses.py
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Tooling configs (ruff, bandit, etc.)
├── .pre-commit-config.yaml
└── README.md             # This file
```

---

## 🚀 Quickstart

1. **Clone this repo:**  
   ```
   git clone https://github.com/wbgreen0405/HuBMAP---Hacking-the-Kidney.git
   ```

2. **Download the competition data** from [Kaggle](https://www.kaggle.com/competitions/hubmap-hacking-the-kidney/data) and place it in the `data/` folder.

3. **Install dependencies and tools:**  
   ```
   pip install -r requirements.txt
   pip install pre-commit
   pre-commit install
   ```

4. **Run code checks locally:**  
   - Lint: `ruff check src/`
   - Security: `bandit -r src/`
   - Tests: `pytest tests/`

5. **Run or explore the main notebooks in `notebooks/` for EDA and model development.**

---

## 🧹 Code Quality & Security

- **Logging:** Scripts use the `logging` module for traceability.
- **Linting:** [ruff](https://github.com/astral-sh/ruff)
- **Security:** [bandit](https://github.com/PyCQA/bandit)
- **Pre-commit:** Automated hooks for formatting, linting, and security.

---

## 📝 Approach & Solution

- **EDA:** See notebooks for data analysis.
- **Modeling:** Deep learning model (e.g., U-Net++/ResNet/efficient segmentation architecture) on FTU masks.
- **Metric:** Dice coefficient as per competition.
- **Validation:** Cross-validation and leaderboard submission.
- **Submission:** Predictions and RLE-encoded masks as per competition.

---

## 📃 References

- [Kaggle competition page](https://www.kaggle.com/competitions/hubmap-hacking-the-kidney)
- [HuBMAP Consortium](https://hubmapconsortium.org/)
- [Diversity Winner Video](https://www.youtube.com/watch?v=pZTjzaP12Sc)

---

## 🙏 Acknowledgements

- Kaggle community, HuBMAP, and all contributors.
