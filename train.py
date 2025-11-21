##treinamento 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(similar_planets["koi_period"], bins=30, kde=True)
plt.xlabel("Orbital Period (days)")
plt.ylabel("Number of Planets")
plt.title("Distribution of Orbital Periods for Super-Earths (1-2 R⊕)")
plt.show()

high_fpp_planets = fpp_table[fpp_table["fpp_prob"] > 0.1]
print(high_fpp_planets[["kepoi_name", "fpp_koi_period", "fpp_prad", "fpp_prob"]])


similar_tess_planets = toi_table[(toi_table["st_rad"] > 1) & (toi_table["st_rad"] < 2)]
print(similar_tess_planets[["toi", "ra", "dec", "st_rad", "st_logg"]])


features = ["koi_period", "koi_time0bk", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad"]

df_train = koi_table[koi_table["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

for col in features:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

df_train = df_train.dropna(subset=features)

X = df_train[features]
y = df_train["koi_disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0) # 1 para Planeta, 0 para Falso

print("Dados prontos para o Random Forest!")
print(f"Tamanho do Treino: {X.shape}")


clf = RandomForestClassifier(n_estimators=100, random_state=42)

kf = KFold(n_splits=5, shuffle=True ,random_state=42)

notas = cross_val_score(clf, X, y, cv=kf, scoring='recall')


print(f"Notas da 5 provas: {notas}")
print(f"Média final do modelo (Recall): {np.mean(notas):.2f}")
print(f"Desvio padrão: {np.std(notas):.2f}")