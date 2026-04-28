#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:53:47 2026

@author: edijsbergholcs
"""

import numpy as np
import random
import os
from deap import base, creator, tools
from ripser import ripser
from persim import PersistenceImager
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ==========================================
# 1. POSMS: PAMATA ĢENĒTISKAIS ALGORITMS
# ==========================================

# Optimizācijas funkcijas
def sphere(individual):
    return sum(x**2 for x in individual),

def rastrigin(individual):
    A = 10
    return A * len(individual) + sum(x**2 - A * np.cos(2 * np.pi * x) for x in individual),

def rosenbrock(individual):
    return sum(100 * (individual[i+1] - individual[i]**2)**2 + (1 - individual[i])**2 for i in range(len(individual)-1)),

def michalewicz(individual, m=10):
    return -sum(np.sin(x) * np.sin((i+1)*x**2 / np.pi)**(2*m) for i, x in enumerate(individual)),

def schaffer(individual):
    return sum(0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1.0 + 0.001 * (x**2 + y**2))**2 for x, y in zip(individual[:-1], individual[1:])),

def alpine(individual):
    return sum(abs(x * np.sin(x) + 0.1 * x) for x in individual),

def composite_function_1(individual):
    f_R = rastrigin(individual)[0]
    f_M = michalewicz(individual)[0]
    f_A = alpine(individual)[0]
    return 0.5 * f_R + 30 * f_M + 10 * f_A,

def composite_function_2(individual):
    OPT_SHIFT = -2
    
    shifted_individual = [x - OPT_SHIFT for x in individual]
    
    f_R = rastrigin(shifted_individual)[0]
    f_S = schaffer(shifted_individual)[0]
    f_A = alpine(shifted_individual)[0]
    return 0.5 * f_R + 30 * f_S * f_A,

# Pielāgotas operātoru funkcijas
def mutUniformFloat(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(low, up)
            
    return individual,

def sus_minimization(individuals, k):
    fitnesses = [ind.fitness.values[0] for ind in individuals]
    max_fit = max(fitnesses)
    
    # Apgriežam vērtības un pievienojam epsilon (1e-6), lai novērstu nulles un negatīvus skaitļus
    positive_fits = [max_fit - fit + 1e-6 for fit in fitnesses]
    
    sum_fits = sum(positive_fits)
    distance = sum_fits / k
    start = random.uniform(0, distance)
    pointers = [start + i * distance for i in range(k)]
    
    selected = []
    i = 0
    current_sum = positive_fits[i]
    
    for p in pointers:
        while current_sum < p:
            i += 1
            # Drošības mehānisms pret peldošā komata (float) noapaļošanas kļūdām
            if i >= len(individuals):
                i = len(individuals) - 1
                break
            current_sum += positive_fits[i]
        selected.append(individuals[i])
        
    return selected

# DEAP ietvara inicializācija
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
DIMENSIONS = 10
POP_SIZE = 100

# Vienmērīgais sadalījums inicializācijai
toolbox.register("attr_float", random.uniform, -5.12, 5.12) 
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIMENSIONS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operatori
toolbox.register("evaluate_sphere", sphere)
toolbox.register("evaluate_rastrigin", rastrigin)
toolbox.register("evaluate_rosenbrock", rosenbrock)
toolbox.register("evaluate_michalewicz", michalewicz)
toolbox.register("posite", composite_function_2)
toolbox.register("select", sus_minimization)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", mutUniformFloat, low=-5.12, up=5.12, indpb=0.1)

# ==========================================
# 2. POSMS: STĀVOKĻU DATU KOPU ĢENERĒŠANA
# ==========================================

# def generate_state_data(state_type, function, gen, seed_val):
#     random.seed(seed_val)
#     pop = toolbox.population(n=POP_SIZE)
    
#     if state_type == "healthy":
#         cxpb, mutpb = 0.7, 0.05
#         target_gen = gen
#     elif state_type == "premature":
#         cxpb, mutpb = 0.7, 0.001
#         target_gen = "stagnation_50"
#     elif state_type == "wandering":
#         cxpb, mutpb = 0.1, 0.9
#         target_gen = 10
        
#     if function == "sphere":
#         fitnesses = list(map(toolbox.evaluate_sphere, pop))
#     elif function == "rastrigin":
#         fitnesses = list(map(toolbox.evaluate_rastrigin, pop))
#     elif function == "rosenbrock":
#         fitnesses = list(map(toolbox.evaluate_rosenbrock, pop))
#     else:
#         fitnesses = list(map(toolbox.evaluate_michalewicz, pop))
        
#     for ind, fit in zip(pop, fitnesses):
#         ind.fitness.values = fit
        
    
#     stagnation_counter = 0
#     best_fitness_history = []
    
#     gen = 0
#     while True:
#         # Fiksēt labāko derīguma vērtību šajā paaudzē
#         current_best = tools.selBest(pop, 1)[0].fitness.values[0]
        
#         # Pārbaudīt apstāšanās kritērijus pirms jaunas paaudzes ģenerēšanas
#         if target_gen == "stagnation_50":
#             if len(best_fitness_history) > 0 and current_best >= best_fitness_history[-1]:
#                 stagnation_counter += 1
#             else:
#                 stagnation_counter = 0
                
#             if stagnation_counter >= 50:
#                 break # Iestājusies pāragra konverģence
#         else:
#             if gen == target_gen:
#                 break # Sasniegta noteiktā apstāšanās paaudze

#         best_fitness_history.append(current_best)

#         elites = list(map(toolbox.clone, tools.selBest(pop, 2)))

#         offspring = toolbox.select(pop, len(pop) - 2)
#         offspring = list(map(toolbox.clone, offspring))

#         # Krustošana
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < cxpb:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values

#         # Mutācija
#         for mutant in offspring:
#             if random.random() < mutpb:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values

#         # Novērtēt indivīdus, kuriem ir nomainījušies gēni
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
#         if function == "sphere":
#             fitnesses = map(toolbox.evaluate_sphere, invalid_ind)
#         elif function == "rastrigin":
#             fitnesses = map(toolbox.evaluate_rastrigin, invalid_ind)
#         elif function == "rosenbrock":
#             fitnesses = map(toolbox.evaluate_rosenbrock, invalid_ind)
#         else:
#             fitnesses = map(toolbox.evaluate_michalewicz, invalid_ind)
            
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit

#         # Jaunā paaudze pilnībā aizstāj veco
#         pop[:] = elites + offspring
#         gen += 1
        
#     print(best_fitness_history[-1])

#     return np.array(pop)

# output_folder = "/Users/edijsbergholcs/Documents/point_clouds"
# os.makedirs(output_folder, exist_ok=True)

# states = ["healthy", "premature", "wandering"]
# functions = ["sphere", "rastrigin", "rosenbrock", "michalewicz"]
# samples_per_func = 250 # 250 paraugi * 4 funkcijas = 1000 paraugi vienam stāvoklim

# healthy_gens = [5, 10, 20, 50, 100]

# seed_counter = 0

# print("Uzsākta datu ģenerēšana. Tas var aizņemt laiku...")

# for state in states:
#     print(f"Ģenerē datus stāvoklim: {state}...")
#     for func in functions:
        
#         # Izveidojam garantētu sadalījumu veselīgajam stāvoklim šai funkcijai
#         # Rezultātā būs saraksts ar 250 elementiem (50 no katras paaudzes)
#         if state == "healthy":
#             func_target_gens = healthy_gens * (samples_per_func // len(healthy_gens))
#             random.seed(42 + functions.index(func)) 
#             random.shuffle(func_target_gens)
            
#         for i in range(samples_per_func):
#             current_seed = seed_counter
#             seed_counter += 1
            
#             # Piešķiram izlozēto paaudzi no sagatavotā saraksta
#             if state == "healthy":
#                 target_gen = func_target_gens[i]
#             else:
#                 target_gen = None 
            
#             point_cloud = generate_state_data(state, func, target_gen, current_seed)
            
#             filename = f"{state}_{func}_{i:03d}.csv"
#             filepath = os.path.join(output_folder, filename)
            
#             np.savetxt(filepath, point_cloud, delimiter=",", fmt="%.6f")

# print(f"Datu ģenerēšana pabeigta. Visi 3000 faili saglabāti mapē '{output_folder}'.")

# ==========================================
# 3. POSMS: TOPOLOĢISKĀ TRANSFORMĀCIJA 
# ==========================================

def compute_tda_features(point_cloud, dim_count):
    diagrams = ripser(point_cloud, maxdim=dim_count-1)['dgms']
    
    max_dist = 33.0 
    
    pimager = PersistenceImager()
    pimager.birth_range = (0.0, max_dist)
    pimager.pers_range = (0.0, max_dist)
    pimager.pixel_size = max_dist / 40.0
    
    vectors = []
    
    for dim in range(dim_count):
        diag = diagrams[dim]
        
        # Filtrējam bezgalību (īpaši aktuāli H0, bet drošības labad atstājam visiem)
        if len(diag) > 0:
            diag = diag[diag[:, 1] != np.inf]
            
        # Ja diagramma ir tukša (tas ir ļoti normāli un bieži notiek ar H1 un H2)
        if len(diag) == 0:
            diag = np.array([[0,0]])
            
        # Ģenerē matricas attēlu konkrētajai dimensijai
        img = pimager.transform(diag)
        
        # Pievienojam sarakstam (katrs ir 1600D)
        vectors.append(img.flatten())
        
    # Apvienojam visus trīs 1600D vektorus vienā garā 4800D vektorā
    vector = np.concatenate(vectors)
    
    return vector

# Mape, kurā glabājas iepriekš ģenerētie dati
input_folder = "/Users/edijsbergholcs/Documents/point_clouds"

X_raw_one = []
X_raw_three = []
y = []

print("Uzsākta punktu mākoņu nolasīšana un TDA aprēķināšana")

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        state_label = filename.split("_")[0]
        
        filepath = os.path.join(input_folder, filename)
        
        # Nolasām punktu mākoni no CSV (katra rinda ir indivīds, katra kolonna - dimensija)
        point_cloud = np.loadtxt(filepath, delimiter=",")
        
        # Izrēķinām TDA iezīmes (4800 dimensiju vektors)
        features_three = compute_tda_features(point_cloud, 3)
        # Paņemam pirmās 1600 dimensijas 1600 dimensiju vektors)
        features_one = features_three[:1600]
        
        X_raw_one.append(features_one)
        X_raw_three.append(features_three)
        y.append(state_label)


x_one = np.array(X_raw_one)
x_three = np.array(X_raw_three)
y = np.array(y)

# Datu standartizācija: transformē datus tā, lai vidējā vērtība būtu 0 un standartnovirze 1
scaler_one = StandardScaler()
scaler_three = StandardScaler()

x_scaled_one = scaler_one.fit_transform(x_one)
x_scaled_three = scaler_three.fit_transform(x_three)

# ==========================================
# 4. POSMS: KLASIFIKĀCIJAS MODEĻU APMĀCĪBA
# ==========================================

def train_and_evaluate_models(X, y_text_labels, dimension_title=""):
    # Pārveidojam teksta klases par skaitļiem
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_text_labels)
    class_names = le.classes_ 
    
    # Stratificēta sadalīšana (80% apmācībai, 20% testēšanai)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    models = {
        "SVM (RBF)": SVC(kernel='rbf'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # best_model = None
    # best_f1 = 0.0
    # best_model_name = ""
    
    # Vārdnīca matricu saglabāšanai pirms zīmēšanas
    saved_conf_matrices = {}
    
    print(f"\nUzsākta modeļu apmācība un novērtēšana ({dimension_title})...\n")
    
    for name, model in models.items():
        start_time = time.time()
        
        # Šķērsvalidācija
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
        
        # apmācība
        model.fit(X_train, y_train)
        
        # Modeļu novērtēšana
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Saglabājam matricas rezultātu grafikiem
        saved_conf_matrices[name] = conf_matrix
        
        elapsed_time = time.time() - start_time
        
        print(f"--- Modelis: {name} ---")
        print(f"CV Makro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Precizitāte: {acc:.4f}, Makro F1: {macro_f1:.4f}")
        print(f"Laiks: {elapsed_time:.2f} sekundes")
        print("-" * 40)
        
    #     if macro_f1 > best_f1:
    #         best_f1 = macro_f1
    #         best_model = model
    #         best_model_name = name
            
    # print(f"\nLabākais modelis integrācijai: {best_model_name} ar Makro F1: {best_f1:.4f}")
    
    # ==========================================
    # PĀRPRATUMU MATRICU VIZUALIZĀCIJA
    # ==========================================
    sns.set_theme(style="white", context="paper", font_scale=0.9) 
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 8)) 
    axes = axes.flatten()
    
    for idx, (name, cm) in enumerate(saved_conf_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=.5, cbar=True, cbar_kws={"shrink": .8})
        
        axes[idx].set_title(f"{name}", fontsize=12, fontweight='bold', pad=8)
        axes[idx].set_xlabel('Prognozētā klase (ML lēmums)', fontsize=10)
        axes[idx].set_ylabel('Patiesā klase (Reālais stāvoklis)', fontsize=10)
        
        axes[idx].tick_params(axis='x', rotation=0, labelsize=9)
        axes[idx].tick_params(axis='y', rotation=0, labelsize=9)

    plt.suptitle(f"Modeļu klasifikācijas precizitāte - {dimension_title}", fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=1.5) 
    
    return models, le

all_trained_models_one, label_encoder = train_and_evaluate_models(x_scaled_one, y, dimension_title="1600D (Tikai Beti-0)")
all_trained_models_three, label_encoder = train_and_evaluate_models(x_scaled_three, y, dimension_title="4800D (Pilna topoloģija)")


# ==========================================
# 5. POSMS: ADAPTĪVĀ ALGORITMA INTEGRĀCIJA
# ==========================================

def adaptive_ga_cycle(best_ml_model, scaler, label_encoder, toolbox, adaptation_strategy="baseline", seed_val=42, dim_count=3):
    random.seed(seed_val)
    pop = toolbox.population(n=POP_SIZE)
    
    fitnesses = list(map(toolbox.posite, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    pm_base, pk_base = 0.05, 0.7
    pm, pk = pm_base, pk_base
    
    pm_min, pm_max = 0.0, 1.0
    pk_min, pk_max = 0.0, 1.0
    
    stagnation_counter = 0
    best_fitness_history = []
    pm_history = []
    pk_history = []
    state_history = []
    fitness = []
    state = "healthy"
    prev_state = "healthy"
    
    start_time = time.time()
    
    for gen in range(1000):
        
        # ==========================================
        # 1. TDA DIAGNOSTIKA
        # ==========================================
        if gen % 5 == 0 and gen > 0 and adaptation_strategy in ["static", "incremental", "hybrid"]:
            features = compute_tda_features(np.array(pop), dim_count)
            features_scaled = scaler.transform([features])
            state_encoded = best_ml_model.predict(features_scaled)
            state = label_encoder.inverse_transform(state_encoded)[0]
            
            if adaptation_strategy == "static":
                if state == "premature": pm, pk = 0.5, 0.5
                elif state == "wandering": pm, pk = 0.01, 0.8
                else: pm, pk = pm_base, pk_base
                    
            elif adaptation_strategy == "incremental":
                delta_m, delta_k = 0.1, 0.1
                if state == "premature":
                    pm = min(pm + delta_m, pm_max)
                    pk = max(pk - delta_k, pk_min)
                elif state == "wandering":
                    pm = max(pm - delta_m, pm_min)
                    pk = min(pk + delta_k, pk_max)
                else:
                    pm = round(pm - 0.01 if pm > pm_base else (pm + 0.01 if pm < pm_base else pm), 2)
                    pk = round(pk - 0.01 if pk > pk_base else (pk + 0.01 if pk < pk_base else pk), 2)
                    
            elif adaptation_strategy == "hybrid":
                if state == "premature":
                    if prev_state == "premature": pm, pk = min(pm + 0.1, pm_max), max(pk - 0.1, pk_min)
                    else: pm, pk = 0.5, 0.5
                elif state == "wandering":
                    if prev_state == "wandering": pm, pk = 0.0, min(pk + 0.05, pk_max)
                    else: pm, pk = 0.01, 0.8
                else: pm, pk = pm_base, pk_base
            
            prev_state = state

        # Fiksē vērtības
        current_best = tools.selBest(pop, 1)[0].fitness.values[0]
        pm_history.append(pm)
        pk_history.append(pk)
        state_history.append(state)
        fitness.append(current_best)
        
        if len(best_fitness_history) > 0 and current_best >= best_fitness_history[-1]:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            
        best_fitness_history.append(current_best)
        if stagnation_counter >= 100:
            break

        # ==========================================
        # 2. LITERATŪRAS AGA PARAMETRU APRĒĶINS
        # ==========================================
        if adaptation_strategy == "literature_aga":
            fits = [ind.fitness.values[0] for ind in pop]
            f_min = min(fits) # Labākā vērtība minimizācijā
            f_avg = sum(fits) / len(fits)
            k1, k3 = 1.0, 1.0 # AGA Krustošanās konstantes
            k2, k4 = 0.5, 0.5 # AGA Mutācijas konstantes
            
        # ==========================================
        # 3. ĢENĒTISKĀS OPERĀCIJAS
        # ==========================================
        elites = list(map(toolbox.clone, tools.selBest(pop, 2)))
        offspring = toolbox.select(pop, len(pop) - 2)
        offspring = list(map(toolbox.clone, offspring))

        # Saglabājam vecāku derīgumu, jo krustošana to izdzēsīs
        parent_fitness = {id(ind): ind.fitness.values[0] for ind in offspring}

        # Krustošana
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if adaptation_strategy == "literature_aga":
                f_prime = min(parent_fitness[id(child1)], parent_fitness[id(child2)])
                if f_prime <= f_avg:
                    pk_current = k1 * (f_prime - f_min) / (f_avg - f_min + 1e-6)
                else:
                    pk_current = k3
            else:
                pk_current = pk 
                
            if random.random() < pk_current:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutācija
        for mutant in offspring:
            if adaptation_strategy == "literature_aga":
                f = parent_fitness[id(mutant)]
                if f <= f_avg:
                    pm_current = k2 * (f - f_min) / (f_avg - f_min + 1e-6)
                else:
                    pm_current = k4
            else:
                pm_current = pm
                
            if random.random() < pm_current:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.posite, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = elites + offspring

    execution_time = time.time() - start_time
    best_ind = tools.selBest(pop, 1)[0]
    
    return best_ind.fitness.values[0], gen, execution_time, pm_history, pk_history, state_history, fitness

# ==========================================
# TESTĒŠANA
# ==========================================

num_runs = 100 
adaptive_strategies = ["static", "incremental", "hybrid"]
comprehensive_results = {}

print("=== UZSĀKTA PAPLAŠINĀTĀ TESTĒŠANA ===")
print("1. Izpilda bāzes GA ...")

baseline_fitness, baseline_gens, baseline_times, baseline_mutation, baseline_crossover, baseline_state, baseline_fitnesses = [], [], [], [], [], [], []
for run in range(num_runs):
    current_seed = 1000 + run
    b_fit, b_gens, b_time, b_pm, b_pk, b_state, b_fitness = adaptive_ga_cycle(None, scaler_one, label_encoder, toolbox, "baseline", current_seed)
    baseline_fitness.append(b_fit)
    baseline_gens.append(b_gens)
    baseline_times.append(b_time)
    baseline_mutation.append(b_pm)
    baseline_crossover.append(b_pk)
    baseline_state.append(b_state)
    baseline_fitnesses.append(b_fitness)

comprehensive_results["BASELINE"] = {
    "fitness": np.array(baseline_fitness), 
    "generations": np.array(baseline_gens), 
    "times": np.array(baseline_times),
    "mutation": np.array(baseline_mutation, dtype=object),
    "crossover": np.array(baseline_crossover, dtype=object),
    "state": np.array(baseline_state, dtype=object),
    "fitnesses": np.array(baseline_fitnesses, dtype=object)
}

print("2. Izpilda klasisko  AGA ...")

literature_fitness, literature_gens, literature_times, literature_mutation, literature_crossover, literature_state, literature_fitnesses = [], [], [], [], [], [], []
for run in range(num_runs):
    current_seed = 1000 + run
    l_fit, l_gens, l_time, l_pm, l_pk, l_state, l_fitness = adaptive_ga_cycle(None, scaler_one, label_encoder, toolbox, "literature_aga", current_seed)
    literature_fitness.append(l_fit)
    literature_gens.append(l_gens)
    literature_times.append(l_time)
    literature_mutation.append(l_pm)
    literature_crossover.append(l_pk)
    literature_state.append(l_state)
    literature_fitnesses.append(l_fitness)

comprehensive_results["LITERATURE_AGA"] = {
    "fitness": np.array(literature_fitness), 
    "generations": np.array(literature_gens), 
    "times": np.array(literature_times),
    "mutation": np.array(literature_mutation, dtype=object),
    "crossover": np.array(literature_crossover, dtype=object),
    "state": np.array(literature_state, dtype=object),
    "fitnesses": np.array(literature_fitnesses, dtype=object)
}

print("3. Uzsāk TDA adaptīvo stratēģiju testēšanu ar 1600D (tikai Beti-0) ML modeļiem...")

for model_name, model in all_trained_models_one.items():
    dict_key = f"{model_name} (1600D)"
    print(f"\n>> Testē modeli kontroliera lomā: {dict_key}")
    comprehensive_results[dict_key] = {}
    
    for strategy in adaptive_strategies:
        print(f"   Izpilda stratēģiju: {strategy.upper()}...", end="", flush=True)
        fit_list, gen_list, time_list, pm_list, pk_list, state_list, fitnesses_list = [], [], [], [], [], [], []
        
        for run in range(num_runs):
            current_seed = 1000 + run 
            fit, gens, exec_time, pm_hist, pk_hist, state_hist, fit_hist = adaptive_ga_cycle(model, scaler_one, label_encoder, toolbox, strategy, current_seed, dim_count=1)
            fit_list.append(fit)
            gen_list.append(gens)
            time_list.append(exec_time)
            pm_list.append(pm_hist)
            pk_list.append(pk_hist)
            state_list.append(state_hist)
            fitnesses_list.append(fit_hist)
            
        comprehensive_results[dict_key][strategy] = {
            "fitness": np.array(fit_list), 
            "generations": np.array(gen_list), 
            "times": np.array(time_list),
            "mutation": np.array(pm_list, dtype=object),
            "crossover": np.array(pk_list, dtype=object),
            "state": np.array(state_list, dtype=object),
            "fitnesses": np.array(fitnesses_list, dtype=object)
        }
        print(" Pabeigts!")
        
print("\n4. Uzsāk TDA adaptīvo stratēģiju testēšanu ar 4800D (Pilnas topoloģijas) ML modeļiem...")

for model_name, model in all_trained_models_three.items():
    dict_key = f"{model_name} (4800D)"
    print(f"\n>> Testē modeli kontroliera lomā: {dict_key}")
    comprehensive_results[dict_key] = {}
    
    for strategy in adaptive_strategies:
        print(f"   Izpilda stratēģiju: {strategy.upper()}...", end="", flush=True)
        fit_list, gen_list, time_list, pm_list, pk_list, state_list, fitnesses_list = [], [], [], [], [], [], []
        
        for run in range(num_runs):
            current_seed = 1000 + run 
            fit, gens, exec_time, pm_hist, pk_hist, state_hist, fit_hist = adaptive_ga_cycle(model, scaler_three, label_encoder, toolbox, strategy, current_seed, dim_count=3)
            fit_list.append(fit)
            gen_list.append(gens)
            time_list.append(exec_time)
            pm_list.append(pm_hist)
            pk_list.append(pk_hist)
            state_list.append(state_hist)
            fitnesses_list.append(fit_hist)
            
        comprehensive_results[dict_key][strategy] = {
            "fitness": np.array(fit_list), 
            "generations": np.array(gen_list), 
            "times": np.array(time_list),
            "mutation": np.array(pm_list, dtype=object),
            "crossover": np.array(pk_list, dtype=object),
            "state": np.array(state_list, dtype=object),
            "fitnesses": np.array(fitnesses_list, dtype=object)
        }
        print(" Pabeigts!")
        
print("\n=== VISI CIKLI VEIKSMĪGI PABEIGTI ===")

# ==========================================
# REZULTĀTU KOPSAVILKUMS
# ==========================================
print("\n" + "="*85)
print(f"{'GALĪGAIS PAPLAŠINĀTĀS TESTĒŠANAS KOPSAVILKUMS':^85}")
print("="*85)

print("\n[ KLASISKĀS METODES (Bez TDA) ]")
for method in ["BASELINE", "LITERATURE_AGA"]:
    if method in comprehensive_results:
        res = comprehensive_results[method]
        print(f"  - {method:<15} | Derīgums: {np.mean(res['fitness']):>8.4f} (Std: {np.std(res['fitness']):>6.4f}) | Paaudzes: {np.mean(res['generations']):>5.1f} | Laiks: {np.mean(res['times']):>5.2f} sek")

ml_models = [k for k in comprehensive_results.keys() if k not in ["BASELINE", "LITERATURE_AGA"]]
ml_models.sort()

for model_key in ml_models:
    print(f"\n[ KONTROLIERIS: {model_key} ]")
    for strategy in adaptive_strategies:
        if strategy in comprehensive_results[model_key]:
            res = comprehensive_results[model_key][strategy]
            print(f"  - {strategy.upper():<12} | Derīgums: {np.mean(res['fitness']):>8.4f} (Std: {np.std(res['fitness']):>6.4f}) | Paaudzes: {np.mean(res['generations']):>5.1f} | Laiks: {np.mean(res['times']):>5.2f} sek")

print("\n" + "="*85 + "\n")

# Saglabājam visus rezultātus failā
with open('results_test_function_2.pkl', 'wb') as f:
    pickle.dump(comprehensive_results, f)

print("Rezultāti saglabāti")


# ==========================================
# GRAFIKS: Siltuma kartes
# ==========================================

states_to_plot = ["healthy", "premature", "wandering"]
state_titles_lv = {
    "healthy": "VESELĪGA ATTĪSTĪBA",
    "premature": "PĀRAGRA KONVERĢENCE",
    "wandering": "KLAIŅOŠANA"
}
colormaps = ['viridis', 'plasma', 'inferno']

sample_indices = {}
for state in states_to_plot:
    idx = np.where(np.array(y) == state)[0][0] 
    sample_indices[state] = idx

fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True, sharey=True)

row_max_v = [0, 0, 0]
matrices = {}

for state in states_to_plot:
    idx = sample_indices[state]
    vector_4800d = x_three[idx]
    
    img_H0 = vector_4800d[0:1600].reshape(40, 40)
    img_H1 = vector_4800d[1600:3200].reshape(40, 40)
    img_H2 = vector_4800d[3200:4800].reshape(40, 40)
    
    matrices[state] = [img_H0, img_H1, img_H2]
    
    row_max_v[0] = max(row_max_v[0], np.max(img_H0))
    row_max_v[1] = max(row_max_v[1], np.max(img_H1))
    row_max_v[2] = max(row_max_v[2], np.max(img_H2))

for dim in range(3):
    for col_idx, state in enumerate(states_to_plot):
        ax = axes[dim, col_idx]
        img = matrices[state][dim]
        
        current_vmax = np.max(img)
        
        if current_vmax == 0:
            current_vmax = 0.001 
            
        im = ax.imshow(img.T, origin='lower', cmap=colormaps[dim], vmin=0, vmax=current_vmax)
        ax.grid(True, color='white', linestyle='-', alpha=0.8, linewidth=0.5, zorder=2)
        
        if dim == 0:
            ax.set_title(state_titles_lv[state], fontsize=13, fontweight='bold', pad=20)
        if col_idx == 0:
            ax.set_ylabel(f"Beti-{dim}\nNoturība", fontsize=12, fontweight='bold')
            
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[2, :]:
    ax.set_xlabel("Dzimšana", fontsize=11)

plt.suptitle(r"Noturīgās homoloģijas siltuma kartes ($\beta_0, \beta_1, \beta_2$)"+ "\n" + "(tieši iegūtas no modeļa apmācības iezīmju matricas)", 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("siltuma_kartes.png", dpi=300)
plt.show()


