import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


#Carregando arquivos

kepler_stellar = pd.read_csv("data/keplerstellar_2025.02.03_04.41.47.csv", comment="#")
stellar_hosts = pd.read_csv("data/STELLARHOSTS_2025.02.03_06.11.17.csv", comment="#")

#arquivo do koy_table
koi_table = pd.read_csv("data/q1_q8_koi_2025.02.03_04.12.15.csv", comment="#")

tce_table = pd.read_csv("data/q1_q17_dr25_tce_2025.02.03_04.32.18.csv", comment="#")


#tess dados possiveis falso positivos
fpp_table = pd.read_csv("data/q1_q17_dr25_koifpp_2025.02.03_06.14.34.csv", comment="#")

#tess confirmados
toi_table = pd.read_csv("data/TOI_2025.02.03_06.18.31.csv", comment="#")


# --- Data Overview ---
print("Kepler Stellar Data:")
print(kepler_stellar.info())
print(kepler_stellar.describe())

print("\nStellar Hosts Data:")
print(stellar_hosts.info())
print(stellar_hosts.describe())

print("\nKOI Data:")
print(koi_table.info())
print(koi_table.describe())

print("\nTCE Data:")
print(tce_table.info())
print(tce_table.describe())


print("\nFPP Data:")
print(fpp_table.info())
print(fpp_table.describe())

print("\nTOI Data:")
print(toi_table.info())
print(toi_table.describe())


#verificando dados/informações base do banco de dados 
print("KOI HEAD", koi_table.head())

#visualização dos exoplanetas periodos orbitais

plt.figure(figsize=(10,5))
sns.histplot(koi_table["koi_period"], bins=50, kde=True, color='red')
plt.xlabel("Obital Period (Days)")
plt.ylabel("Count")

plt.savefig('grafico_periodo_orbital.png')

#Temperatura estelas vs planetas radiano

plt.figure(figsize=(10,5))
sns.scatterplot(data=koi_table, x="koi_steff", y="koi_prad", alpha=0.5)
plt.xlabel("Stellar Effective Temperature ")
plt.ylabel("Planetary Radius")
plt.title("Exoplanet Radius vs Stellar Temperature")

plt.savefig('grafico_.png')


koi_fpp = pd.merge(koi_table, fpp_table, on="kepid", how="left")




import lightkurve as lk

warnings.filterwarnings("ignore", category=UserWarning)

def download_and_plot_lightcurve(kepid, koi_name, koi_period=None):

    try:
        search_result = lk.search_lightcurve(f"KIC{kepid}")

        if len(search_result) == 0:
            search_result = lk.search_lightcurve(f"KIC{kepid}")


        if len(search_result) == 0:
            print(f"No light curve found for KOI {koi_name} (KIC{kepid})")

            return None
        
        lc_collection = search_result.download_all()


        if lc_collection is None:
            print(f"Falha ao realizar o Donwload da curva de luz {koi_name} (KIC {kepid})")
            return None
        

        lc = lc_collection.stitch() if len(lc_collection) > 1 else lc.collection[0]


        lc.plot(title=f"Luz de curva para KOI {koi_name} (KIC{kepid})", xlabel="Tempo (dias)", ylabel="Flux")

        plt.savefig('candidatos.png')

        plt.show()

        if koi_period:
            folded_lc = lc.fold(period=koi_period)
            folded_lc.plot(title=f"000 KOI{koi_name}")

            plt.savefig('candidatos2.png')

        lc_clean = lc.remove_outliers(sigma=5)  # Remove extreme values (5-sigma clipping)
        
        lc_clean.plot(title="Cleaned Light Curve")


        plt.savefig('candidatos_limpo.png')


        return lc
    
    except Exception as e:
        print(f"Erro no processando KOI {koi_name} (KIC {kepid}):{e}")

        return None
    

koi_sample = koi_table[koi_table["koi_disposition"] == "CONFIRMED"].iloc[0]

kepid = koi_sample["kepid"]

koi_name = koi_sample["kepoi_name"]

koi_period = koi_sample["koi_period"]


lc = download_and_plot_lightcurve(kepid, koi_name, koi_period)


from astropy.timeseries import BoxLeastSquares
lc_clean = lc.remove_outliers(sigma=5)  # Remove extreme values (5-sigma clipping)

bls = BoxLeastSquares(lc_clean.time, lc_clean.flux)


periods = np.linspace(0.5, 100, 5000)

bls_power = bls.power(periods, duration = 0.1)

best_period = periods[np.argmax(bls_power.power)]

print(f"Periodo detectado: {best_period:.3f} days")

plt.plot(periods, bls_power.power)

plt.figure(figsize=(12,6))

plt.xlabel("Periodo (dias)")
plt.ylabel("BLS poder")
plt.title("BLS periodograma")

plt.savefig('bls_periodograma.png')




##deixando mais bonito o grafico para visualização

folded_lc = lc_clean.fold(period=best_period)

time_values = folded_lc.time.value

plt.figure(figsize=(10, 5))
plt.plot(time_values, folded_lc.flux, '.', markersize=1, label="Folded Data")
plt.xlabel("Phase (days)")
plt.ylabel("Normalized Flux")
plt.title(f"Folded Light Curve (Period: {best_period:.3f} days)")
plt.legend()

plt.savefig('bls_periodograma_ponto.png')


similar_planets = koi_table[(koi_table["koi_prad"]>1) & (koi_table["koi_prad"]<2)]

print(similar_planets[["kepoi_name", "koi_prad" ,"koi_period"]])

fpp_table[fpp_table["kepid"] == 9388479]

toi_table[toi_table["ra"].round(2) == koi_sample["ra"].round(2)]



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


features = ["koi_period", "koi_time0bk", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad"]

df_train = koi_table[koi_table["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

for col in features:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

df_train = df_train.dropna(subset=features)


X = df_train[features]
y = df_train["koi_disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0) # 1 para Planeta, 0 para Falso

print("Dados prontos para o Random Forest")
print(f"Tamanho do Treino: {X.shape}")


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, recall_score

clf = RandomForestClassifier(n_estimators=100)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
    


notas_recall = cross_val_score(clf, X, y, cv=kf, scoring='recall')
    
notas_precision = cross_val_score(clf, X, y, cv=kf, scoring='precision')


print("\n" + "="*40)
print("RESULTADOS DO K-FOLD (5 PARTES)")
print("="*40)
    
print(f"Notas de Recall nas 5 provas: \n{notas_recall}")
print(f"\n-> MÉDIA DE RECALL: {np.mean(notas_recall):.2%}")
print(f"-> Estabilidade (Desvio Padrão): +/- {np.std(notas_recall):.2%}")
    
print("-" * 40)
print(f"Média de Precisão: {np.mean(notas_precision):.2%}")
print("="*40)

plt.figure(figsize=(8, 5))
data_to_plot = pd.DataFrame({
    'Recall (Não perder planetas)': notas_recall, 
    'Precision (Não dar alarme falso)': notas_precision
})
    
sns.boxplot(data=data_to_plot, palette="Set3")
plt.title("Estabilidade do Modelo: K-Fold (5 splits)")
plt.ylabel("Pontuação (0 a 1)")
plt.grid(True, alpha=0.3)
plt.savefig('estabilidade.png')


clf_final = RandomForestClassifier(n_estimators=100)
clf_final.fit(X,y)

df_candidatos = koi_table[koi_table["koi_disposition"].str.strip() == "CANDIDATE"]

for col in features:
    df_candidatos[col] = pd.to_numeric(df_candidatos[col], errors='coerce')


df_candidatos = df_candidatos.dropna(subset=features)

X_candidatos = df_candidatos[features]

print(f"{len(X_candidatos)} candidatos")



probabilidades = clf_final.predict_proba(X_candidatos)

df_candidatos["chance_de_positivo"] = probabilidades[:,1]

descobertas = df_candidatos.sort_values(by="chance_de_positivo",  ascending=False)

colunas_exibicao = ["kepoi_name", "chance_de_positivo", "koi_period", "koi_prad", "koi_steff"]
print(descobertas[colunas_exibicao].head(10).to_string(index=False))


descobertas.to_csv("sinalizacoes_exoplanetas.csv", index=False)
print("Lista salva")




##anomalias code


iso_forest = IsolationForest(contamination=0.02, random_state=42)

iso_forest.fit(X)

anomalias = iso_forest.predict(X) #1 = normal, -1 = anomalia

scores = iso_forest.decision_function(X)

df_anomalias = df_train.copy()

df_anomalias['anomalia'] = anomalias

df_anomalias['grau_de_anomalia'] = scores

suspeitos_anoma = df_anomalias[df_anomalias['anomalia'] == -1]

suspeitos_anoma = suspeitos_anoma.sort_values(by='grau_de_anomalia')


cols_to_show = ['kepoi_name', 'koi_disposition', 'koi_period', 'koi_prad', 'grau_de_anomalia']
print(suspeitos_anoma[cols_to_show].head(10))

suspeitos_anoma.to_csv("exoplanetas_anomalos.csv", index=False)


plt.figure(figsize=(12, 6))

plt.scatter(df_anomalias[df_anomalias['anomalia'] == 1]['koi_period'], 
            df_anomalias[df_anomalias['anomalia'] == 1]['koi_prad'], 
            c='blue', s=10, alpha=0.3, label='Normal')

plt.scatter(suspeitos_anoma['koi_period'], 
            suspeitos_anoma['koi_prad'], 
            c='red', s=50, marker='x', label='Anomalia')

plt.xscale('log') # Escala logarítmica ajuda a ver melhor dados astronomicos
plt.yscale('log')
plt.xlabel('Período Orbital (Dias)')
plt.ylabel('Raio do Planeta (Raios Terrestres)')
plt.title('Anomalias detectadas pelo Isolation Forest')
plt.legend()
plt.show()

plt.savefig('estranhos.png')


