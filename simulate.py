"""
================================================
PUV Arrival Time Estimation — LNU Route
Robinsons Marasbaras → Leyte Normal University
================================================
Run this file with:   python simulate.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH    = "data/rob_marasbaras_to_lnu_fixed.csv"
OUTPUT_DIR  = "outputs"
ROUTE_KM    = 3.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  PUV ETA Simulation — Marasbaras to LNU")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/5] Loading data...")
df = pd.read_csv(CSV_PATH)
df = df[['Trip_ID','Entry_Time','LTI_Mean','LTI_Max',
          'Velocity_kmh','Rush_Hour','Weather','Travel_Time_Min']].dropna()
print(f"      Loaded {len(df)} trips.")

# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2/5] Engineering features...")
df['Cumulative_Dist_Ratio'] = (
    df['Velocity_kmh'] * (df['Travel_Time_Min'] / 60) / ROUTE_KM
).clip(upper=1.0)

FEATURES = ['LTI_Mean', 'LTI_Max', 'Velocity_kmh',
            'Rush_Hour', 'Weather', 'Cumulative_Dist_Ratio']
TARGET   = 'Travel_Time_Min'

X = df[FEATURES]
y = df[TARGET]
print(f"      Features: {FEATURES}")

# ─────────────────────────────────────────────
# STEP 3: SPLIT — 40 train, 10 hold-out
# ─────────────────────────────────────────────
print("\n[3/5] Splitting dataset (40 train / 10 test)...")
X_train, X_test = X.iloc[:40], X.iloc[40:]
y_train, y_test = y.iloc[:40], y.iloc[40:]

# ─────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[4/5] Training models...")

rfr = RandomForestRegressor(n_estimators=100, max_depth=5,
                             min_samples_split=4, random_state=42)
rfr.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rfr_cv = cross_val_score(rfr, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
lr_cv  = cross_val_score(lr,  X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print("      Random Forest trained.")
print("      Linear Regression trained.")

# ─────────────────────────────────────────────
# STEP 5: EVALUATE
# ─────────────────────────────────────────────
print("\n[5/5] Evaluating on 10 hold-out trips...")

rfr_preds = rfr.predict(X_test)
lr_preds  = lr.predict(X_test)

def show_metrics(name, actual, preds, cv_scores):
    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape = mean_absolute_percentage_error(actual, preds) * 100
    cv_mae = -cv_scores.mean()
    print(f"\n  ── {name} ──")
    print(f"     MAE  : {mae:.2f} min  ← avg prediction error")
    print(f"     RMSE : {rmse:.2f} min  ← penalizes big errors")
    print(f"     MAPE : {mape:.1f} %   ← % error")
    print(f"     CV MAE (10-fold): {cv_mae:.2f} min")
    return mae, rmse, mape

rfr_mae, rfr_rmse, rfr_mape = show_metrics("Random Forest Regressor (Main)",  y_test, rfr_preds, rfr_cv)
lr_mae,  lr_rmse,  lr_mape  = show_metrics("Linear Regression (Baseline)",     y_test, lr_preds,  lr_cv)

# Feature importance
fi = pd.Series(rfr.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  ── Feature Importance (Random Forest) ──")
for feat, val in fi.items():
    bar = "█" * int(val * 40)
    print(f"     {feat:<25} {val:.3f}  {bar}")

# ─────────────────────────────────────────────
# SCENARIO SIMULATOR
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SCENARIO SIMULATOR")
print("=" * 55)

def predict_eta(departure_str, lti_mean, lti_max, velocity, rush_hour, weather, dist_ratio=1.0):
    feat = [[lti_mean, lti_max, velocity, rush_hour, weather, dist_ratio]]
    travel_min = rfr.predict(feat)[0]
    dep = datetime.strptime(departure_str, "%H:%M")
    eta = dep + timedelta(minutes=travel_min)
    return round(travel_min, 1), eta.strftime("%I:%M %p")

scenarios = [
    ("Rush hour + Heavy Rain",   "07:00", 5, 5, 3.27, 1, 4),
    ("Rush hour + Clear",        "07:00", 3, 4, 4.80, 1, 1),
    ("Off-peak + Clear",         "10:00", 2, 2, 6.97, 0, 1),
    ("Off-peak + Cloudy",        "13:00", 2, 3, 7.20, 0, 2),
    ("PM Rush + Clear",          "17:00", 4, 5, 4.32, 1, 1),
    ("Best Case (Free Flow)",    "09:00", 1, 1, 14.40, 0, 1),
]

print(f"\n  {'Scenario':<28} {'Travel Time':>12}  {'ETA at LNU':>12}")
print("  " + "-" * 56)
for label, dep, lm, lx, vel, rh, wx in scenarios:
    t, eta = predict_eta(dep, lm, lx, vel, rh, wx)
    print(f"  {label:<28} {t:>9.1f} min  {eta:>12}")

# ─────────────────────────────────────────────
# SAVE CHARTS
# ─────────────────────────────────────────────
print("\n  Saving charts to outputs/ ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PUV ETA Simulation — Marasbaras to LNU", fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#F8F9FA')

# Chart 1: Actual vs Predicted
ax = axes[0, 0]
trips = range(1, 11)
ax.plot(trips, y_test.values, 'o-', color='#2563EB', lw=2, ms=7, label='Actual')
ax.plot(trips, rfr_preds,    's--', color='#16A34A', lw=2, ms=7, label='RFR Predicted')
ax.plot(trips, lr_preds,     '^:',  color='#EA580C', lw=2, ms=7, label='LR Predicted')
ax.set_title('Actual vs Predicted (10 Hold-out Trips)', fontweight='bold')
ax.set_xlabel('Trip #'); ax.set_ylabel('Travel Time (min)')
ax.legend(); ax.set_xticks(list(trips)); ax.grid(alpha=0.3); ax.set_facecolor('white')

# Chart 2: MAE comparison
ax = axes[0, 1]
models_names = ['Random Forest', 'Linear Regression']
maes = [rfr_mae, lr_mae]
colors = ['#16A34A', '#EA580C']
bars = ax.bar(models_names, maes, color=colors, alpha=0.85, edgecolor='white', width=0.4)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f} min', ha='center', fontweight='bold')
ax.set_title('MAE Comparison (Lower = Better)', fontweight='bold')
ax.set_ylabel('MAE (minutes)'); ax.grid(alpha=0.3, axis='y'); ax.set_facecolor('white')

# Chart 3: Feature Importance
ax = axes[1, 0]
bar_colors = ['#16A34A' if v > 0.1 else '#2563EB' if v > 0.01 else '#9CA3AF' for v in fi.values]
ax.barh(fi.index, fi.values, color=bar_colors, alpha=0.85, edgecolor='white')
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance Score'); ax.grid(alpha=0.3, axis='x'); ax.set_facecolor('white')

# Chart 4: Scenario bar chart
ax = axes[1, 1]
s_labels = [s[0] for s in scenarios]
s_times  = [predict_eta(*s[1:])[0] for s in scenarios]
s_colors = ['#DC2626','#EA580C','#16A34A','#16A34A','#EA580C','#2563EB']
ax.barh(s_labels, s_times, color=s_colors, alpha=0.85, edgecolor='white')
ax.axvline(df['Travel_Time_Min'].mean(), color='black', lw=1.5, linestyle='--',
           label=f"Avg: {df['Travel_Time_Min'].mean():.1f} min")
ax.set_title('Scenario Simulation', fontweight='bold')
ax.set_xlabel('Predicted Travel Time (min)')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x'); ax.set_facecolor('white')

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "simulation_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Chart saved → {chart_path}")

# Save hold-out results CSV
results_df = df.iloc[40:][['Trip_ID','Travel_Time_Min']].copy()
results_df['RFR_Predicted'] = rfr_preds.round(1)
results_df['RFR_Error_min'] = (rfr_preds - y_test.values).round(1)
results_df['LR_Predicted']  = lr_preds.round(1)
results_df['LR_Error_min']  = (lr_preds - y_test.values).round(1)
csv_path = os.path.join(OUTPUT_DIR, "holdout_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Results saved → {csv_path}")

print("\n" + "=" * 55)
print("  DONE! Check the outputs/ folder.")
print("=" * 55)
