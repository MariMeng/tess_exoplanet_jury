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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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