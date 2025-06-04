# Perovskite Solar Cell Stability Prediction

This project predicts the **stability (T80)** of perovskite solar cells using machine learning techniques. The pipeline includes **data preprocessing**, **feature selection**, **model training with XGBoost**, and **interpretability analysis using SHAP**.

---

##  Project Structure
Perovskite_stability/
├── data/ # Raw dataset (Perovskite.csv)
├── notebooks/ # Jupyter notebook for exploratory analysis
├── scripts/ # Python scripts for preprocessing and model training
├── models/ # Trained models and preprocessing pipelines
├── results/ # Output figures and evaluation metrics
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SirHarginger/Perovskite_stability.git
   cd Perovskite_stability


2. Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

3. Install the required dependencies:
pip install -r requirements.txt

4. Download Perovskite.csv and place it in the data/ directory.



## How to Run
1. pip install -r requirements.txt

2. python scripts/preprocess.py

3. python scripts/train_model.py

4. jupyter notebook notebooks/Perovskite_stability_analysis.ipynb






