#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:39:31 2026

@author: edijsbergholcs
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Patch


# filename = ["ga_comprehensive_results_v3.pkl", 1]
filename = ["results_test_function_2.pkl", 2]


print(f"Ielādē datus no faila: {filename[0]}...")
with open(filename[0], 'rb') as f:
    results = pickle.load(f)

rows = []

for key, data in results.items():
    if key in ["BASELINE", "LITERATURE_AGA"]:
        fits = data['fitness']
        times = data['times']
        gens = data['generations']
        for topo in ['1600D (Tikai $\\beta_0$)', '4800D ($\\beta_0$, $\\beta_1$, $\\beta_2$)']:
            for f_val, t_val, g_val in zip(fits, times, gens):
                rows.append({
                    "Konfigurācija": "Bāzes GA" if key == "BASELINE" else "Adaptīvais GA",
                    "Pilns_Nosaukums": "Bāzes GA\n(Nav TDA)" if key == "BASELINE" else "Adaptīvais GA\n(Nav TDA)",
                    "Derīgums": f_val,
                    "Laiks": t_val,
                    "Paaudzes": g_val,
                    "Topoloģija": topo,
                    "Stratēģija": "N/A"
                })
    else:
        if "(1600D)" in key:
            topo = '1600D (Tikai $\\beta_0$)'
            base_model = key.replace(" (1600D)", "")
        else:
            topo = '4800D ($\\beta_0$, $\\beta_1$, $\\beta_2$)'
            base_model = key.replace(" (4800D)", "")
            
        for strategy, strat_data in data.items():
            fits = strat_data['fitness']
            times = strat_data['times']
            gens = strat_data['generations']
            
            # Īso nosaukumu ģenerēšana
            strat_short = {"static": "Stat.", "incremental": "Inkr.", "hybrid": "Hibr."}[strategy]
            model_short = base_model.replace("Random Forest", "RF").replace("XGBoost", "XGB").replace("SVM (RBF)", "SVM")
            full_name = f"{model_short}\n({strat_short})"
            
            for f_val, t_val, g_val in zip(fits, times, gens):
                rows.append({
                    "Konfigurācija": f"{model_short} ({strat_short})", 
                    "Pilns_Nosaukums": full_name,
                    "Derīgums": f_val,
                    "Laiks": t_val,
                    "Paaudzes": g_val,
                    "Topoloģija": topo,
                    "Stratēģija": strategy
                })

df = pd.DataFrame(rows)
print("Datu tabula sagatavota veiksmīgi!")
df['Laiks_Paaudzē'] = df['Laiks'] / df['Paaudzes'] 

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# GRAFIKS 1: PILNAIS ABLĀCIJAS BOXPLOT (VISI MODELI)
# ==========================================
print("Zīmē grafiku 1 (Pilnais boxplot)...")
plt.figure(figsize=(18, 8))

sns.boxplot(
    data=df, x='Pilns_Nosaukums', y='Derīgums', hue='Topoloģija', 
    palette='Set1', linewidth=1.2, fliersize=3,
    showfliers=False  
)
if filename[1] == 1:
    pass
else:
    plt.ylim(-1, 20) 

plt.title("Risinājuma kvalitātes sadalījums visiem konfigurācijas variantiem", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Labākais derīgums (zemāks ir labāks)", fontsize=13)
plt.xlabel("Algoritma konfigurācija", fontsize=13)
plt.xticks(rotation=0, fontsize=11)
plt.legend(title="Ievades iezīmes modeļiem", title_fontsize='12', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"boxplot_{filename[1]}.png", dpi=300)
plt.show()

boxplot_stats = df.groupby(["Konfigurācija", "Topoloģija"])["Derīgums"].describe()

boxplot_stats = boxplot_stats.rename(columns={
    'count': 'Eksperimentu_skaits',
    'mean': 'Vidējā vērtība',
    'std': 'Standartnovirze',
    'min': 'Minimums',
    '25%': 'Q1',
    '50%': 'Mediāna',
    '75%': 'Q3',
    'max': 'Maksimums'
}).reset_index()

csv_filename = f"boxplot_kopsavilkums_{filename[1]}.csv"

boxplot_stats.to_csv(csv_filename, index=False, encoding='utf-8-sig')


# ==========================================
# GRAFIKS 2: LAIKS PRET KVALITĀTI
# ==========================================
print("Zīmē grafiku 2 (Laiks pret kvalitāti)...")
plt.figure(figsize=(12, 7))

summary_df = df.groupby(["Konfigurācija", "Topoloģija"]).agg({"Derīgums": "mean", "Laiks_Paaudzē": "mean"}).reset_index()
summary_df.to_csv(f"laiks_derigums_{filename[1]}.csv", index=False, encoding='utf-8-sig')

sns.scatterplot(
    data=summary_df, 
    x="Laiks_Paaudzē",
    y="Derīgums", 
    hue="Konfigurācija", 
    style="Topoloģija", 
    s=150, palette="tab20", alpha=0.8, edgecolor='black', linewidth=1
)

if filename[1] == 1:
    plt.ylim(-90, -70) 
else:
    plt.ylim(3, 8) 

plt.title("Skaitļošanas laiks pret risinājuma kvalitāti", fontsize=15, pad=15)
plt.xlabel("Vidējais izpildes laiks vienai paaudzei (sekundes)", fontsize=13)
plt.ylabel("Vidējais labākais derīgums (zemāks ir labāks)", fontsize=13)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(f"scatter_{filename[1]}.png", dpi=300)
plt.show()

# ==========================================
# GRAFIKS 3: PARAMETRU ADAPTĀCIJAS LĪKNES (NO REĀLA CIKLA)
# ==========================================
print("Zīmē grafiku 3 (Parametru adaptācijas līkne)...")
target_model = "SVM (RBF) (4800D)"

if target_model in results:
    panels = [
        ("BASELINE", None, "A. Bāzes Ģenētiskais Algoritms"),
        ("LITERATURE_AGA", None, "B. Adaptīvais Ģenētiskais Algoritms"),
        (target_model, "static", "C. TDA: Statiskā pieeja"),
        (target_model, "incremental", "D. TDA: Inkrementālā pieeja"),
        (target_model, "hybrid", "E. TDA: Hibrīdā pieeja")
    ]
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    
    color_pm = "#1f77b4" # Zils
    color_pk = "#d62728" # Sarkans

    max_gens_all = 0
    gen = 81

    for idx, (model_key, strat_key, base_title) in enumerate(panels):
        ax = axes[idx]
        
        if strat_key is None:
            pm_hist = results[model_key]['mutation'][gen]
            pk_hist = results[model_key]['crossover'][gen]
            state_hist = results[model_key]['state'][gen]
            total_gens = results[model_key]['generations'][gen]
            fit_val = results[model_key]['fitness'][gen]
        else:
            pm_hist = results[model_key][strat_key]['mutation'][gen]
            pk_hist = results[model_key][strat_key]['crossover'][gen]
            state_hist = results[model_key][strat_key]['state'][gen]
            total_gens = results[model_key][strat_key]['generations'][gen]
            fit_val = results[model_key][strat_key]['fitness'][gen]
            
        if total_gens > max_gens_all:
            max_gens_all = total_gens
        
        # fona krāsas 
        for g, st in enumerate(state_hist):
            if st == "healthy": c = '#2ecc71'
            elif st == "premature": c = '#e74c3c'
            elif st == "wandering": c = '#f39c12'
            else: c = 'grey'
            ax.axvspan(g, g+1, color=c, alpha=0.15, linewidth=0)
            
        ax.step(range(len(pm_hist)), pm_hist, color=color_pm, linewidth=2.5, where='post')
        ax.step(range(len(pk_hist)), pk_hist, color=color_pk, linewidth=2.5, linestyle="--", where='post')
        
        final_title = f"{base_title} | Gala derīgums: {fit_val:.4f}"
        
        ax.set_title(final_title, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylabel("Varbūtība")
        ax.set_ylim(-0.05, 1.05) 

    # Uzstādām X asij limitu, lai visi grafiki iet līdz garākajam ciklam
    for ax in axes:
        ax.set_xlim(0, max_gens_all)

    axes[4].set_xlabel("Paaudze", fontsize=13)

    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.3, label='Normāla attīstība'),
        Patch(facecolor='#e74c3c', alpha=0.3, label='Pāragra konverģence'),
        Patch(facecolor='#f39c12', alpha=0.3, label='Klaiņošana'),
        Line2D([0], [0], color=color_pm, lw=2.5, label='Mutācija ($p_m$)'),
        Line2D([0], [0], color=color_pk, lw=2.5, linestyle="--", label='Krustošanās ($p_k$)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=12)

    plt.suptitle("Parametru dinamikas salīdzinājums: Klasiskās metodes pret TDA kontrolieriem", 
                 fontsize=16, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    plt.savefig(f"run_visual_{filename[1]}.png", dpi=300)
    plt.show()
