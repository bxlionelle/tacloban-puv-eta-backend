# PUV Arrival Time Estimation — LNU Route
### Robinsons Marasbaras → Leyte Normal University, Tacloban City

---

## 📁 Folder Structure

```
puv_eta_simulation/
│
├── data/
│   └── rob_marasbaras_to_lnu_fixed.csv   ← your dataset
│
├── outputs/                               ← results go here (auto-created)
│   ├── simulation_results.png
│   └── holdout_results.csv
│
├── simulate.py                            ← MAIN script to run
├── requirements.txt                       ← libraries needed
└── README.md                              ← this file
```

---

## 🖥️ How to Run (Step by Step)

### Step 1 — Make sure Python is installed
Open CMD and type:
```
python --version
```
You should see something like `Python 3.10.x`. If not, download Python from https://python.org

---

### Step 2 — Go into the project folder
```
cd puv_eta_simulation
```

---

### Step 3 — Install the required libraries (only once)
```
pip install -r requirements.txt
```

---

### Step 4 — Run the simulation
```
python simulate.py
```

---

## ✅ What You Will See

```
=======================================================
  PUV ETA Simulation — Marasbaras to LNU
=======================================================

[1/5] Loading data...        ← reads your CSV
[2/5] Engineering features...← builds LTI, velocity, rush hour, etc.
[3/5] Splitting dataset...   ← 40 trips train, 10 trips test
[4/5] Training models...     ← trains Random Forest + Linear Regression
[5/5] Evaluating...          ← shows MAE, RMSE, MAPE

── Random Forest (Main) ──
   MAE  : 0.70 min
   RMSE : 1.10 min
   MAPE : 2.6 %

── Scenario Simulator ──
   Rush hour + Heavy Rain     →  61 min  →  ETA: 08:01 AM
   Best Case (Free Flow)      →  16 min  →  ETA: 09:16 AM
   ...
```

Then check the **outputs/** folder for:
- `simulation_results.png` — all charts
- `holdout_results.csv` — predicted vs actual table

---

## 📊 What the Metrics Mean (Plain English)

| Metric | What it means |
|--------|---------------|
| **MAE** | On average, how many minutes off the prediction is |
| **RMSE** | Same but punishes big mistakes harder |
| **MAPE** | The error as a percentage (e.g., 2.6% = very accurate) |

---

## ❓ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `python not found` | Install Python from python.org |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError` | Make sure you are inside the `puv_eta_simulation/` folder |
| Charts not opening | Check the `outputs/` folder — they're saved there |
