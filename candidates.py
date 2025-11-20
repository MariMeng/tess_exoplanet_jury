import lightkurve as lk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def download_and_plot_lightcurve(kepid, koi_name, koi_period=None):

    try:
        search_result = lk.search_lightcurve(koi_name)

        if len(search_result) == 0:
            search_result = lk.search_lightcurve(f"KIC{kepid}")


        if len(search_result) == 0:
            print(f"No light curve found for KOI {koi_name} (KIC{kepid})")

            return None
        
        lc_collection = search_result.download_all()


        if lc_collection is None:
            print(f"Falha ao realizar o Donwload da curva de luz {koi_name} (KIC {kepid})")
            return None
        

        lc = lc_collection


        lc.plot(title=f"Luz de curva para KOI {koi_name} (KIC{kepid})", xlabel="Tempo (dias)", ylabel="Flux")

        plt.savefig('candidatos.png')

        plt.show()



        return lc
    
    except Exception as e:
        print(f"Erro no processando KOI {koi_name} (KIC {kepid}):{e}")

        return None
    

koi_sample = koi_table[koi_table["koi_disposition"] == "CONFIRMED"].iloc[0]

kepid = koi_sample["kepid"]

koi_name = koi_sample["kepoi_name"]

koi_period = koi_sample["koi_period"]


lc = download_and_plot_lightcurve(kepid, koi_name, koi_period)