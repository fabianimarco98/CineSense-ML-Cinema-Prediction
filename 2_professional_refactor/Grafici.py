import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr
import os


# Directory del file corrente
output_dir = os.path.join(os.path.dirname(__file__), "grafici")

# Crea la cartella se non esiste
os.makedirs(output_dir, exist_ok=True)

#### PRE-PROCESSING PER CREAZIONE GRAFICI ####

path = "input/dati_cinema_updated.xlsx"
df = pd.read_excel(path)
#print(df.head)


# Raggruppo per mese e somma gli ingressi totali
ingressi_mensili_totali = df.groupby(pd.Grouper(key="datetime", freq="ME"))[["total"]].sum().reset_index()
# Creo colonna con mese e anno in italiano (es. 'Mar 2023')
ingressi_mensili_totali["mese_anno"] = ingressi_mensili_totali["datetime"].dt.strftime("%b %Y")
# Ordine cronologico corretto
ingressi_mensili_totali["mese_anno"] = pd.Categorical(ingressi_mensili_totali["mese_anno"], categories=ingressi_mensili_totali["mese_anno"], ordered=True)
#print(ingressi_mensili_totali)

# Raggruppo per anno e somma gli ingressi totali
ingressi_annuali_totali = df.groupby(pd.Grouper(key="datetime", freq="Y"))[["total"]].sum().reset_index()
# Creo colonna con anno (es. '2023')
ingressi_annuali_totali["anno"] = ingressi_annuali_totali["datetime"].dt.strftime("%Y")
# Ordine cronologico corretto
ingressi_annuali_totali["anno"] = pd.Categorical(ingressi_annuali_totali["anno"],categories=ingressi_annuali_totali["anno"],ordered=True)
#print(ingressi_annuali_totali)


# Raggruppo per mese e somma gli ingressi full_price
ingressi_mensili_full_price = df.groupby(pd.Grouper(key="datetime", freq="ME"))[["full_price"]].sum().reset_index()
# Creo colonna con mese e anno in italiano (es. 'Mar 2023')
ingressi_mensili_full_price["mese_anno"] = ingressi_mensili_full_price["datetime"].dt.strftime("%b %Y")
# Ordine cronologico corretto
ingressi_mensili_full_price["mese_anno"] = pd.Categorical(ingressi_mensili_full_price["mese_anno"], categories=ingressi_mensili_full_price["mese_anno"], ordered=True)
#print(ingressi_mensili_full_price)

# Raggruppo per anno e somma gli ingressi full_price
ingressi_annuali_full_price = df.groupby(pd.Grouper(key="datetime", freq="Y"))[["full_price"]].sum().reset_index()
# Creo colonna con anno (es. '2023')
ingressi_annuali_full_price["anno"] = ingressi_annuali_full_price["datetime"].dt.strftime("%Y")
# Ordine cronologico corretto
ingressi_annuali_full_price["anno"] = pd.Categorical(ingressi_annuali_full_price["anno"],categories=ingressi_annuali_full_price["anno"],ordered=True)
#print(ingressi_annuali_full_price)


# Raggruppo per mese e somma gli ingressi reduced
ingressi_mensili_reduced = df.groupby(pd.Grouper(key="datetime", freq="ME"))[["reduced"]].sum().reset_index()
# Creo colonna con mese e anno in italiano (es. 'Mar 2023')
ingressi_mensili_reduced["mese_anno"] = ingressi_mensili_reduced["datetime"].dt.strftime("%b %Y")
# Ordine cronologico corretto
ingressi_mensili_reduced["mese_anno"] = pd.Categorical(ingressi_mensili_reduced["mese_anno"], categories=ingressi_mensili_reduced["mese_anno"], ordered=True)
#print(ingressi_mensili_reduced)


# Raggruppo per anno e somma gli ingressi reduced
ingressi_annuali_reduced = df.groupby(pd.Grouper(key="datetime", freq="Y"))[["reduced"]].sum().reset_index()
# Creo colonna con anno (es. '2023')
ingressi_annuali_reduced["anno"] = ingressi_annuali_reduced["datetime"].dt.strftime("%Y")
# Ordine cronologico corretto
ingressi_annuali_reduced["anno"] = pd.Categorical(ingressi_annuali_reduced["anno"],categories=ingressi_annuali_reduced["anno"],ordered=True)
#print(ingressi_annuali_reduced)


# Raggruppo per mese e somma gli ingressi free
ingressi_mensili_free = df.groupby(pd.Grouper(key="datetime", freq="ME"))[["free"]].sum().reset_index()
# Creo colonna con mese e anno in italiano (es. 'Mar 2023')
ingressi_mensili_free["mese_anno"] = ingressi_mensili_free["datetime"].dt.strftime("%b %Y")
# Ordine cronologico corretto
ingressi_mensili_free["mese_anno"] = pd.Categorical(ingressi_mensili_free["mese_anno"], categories=ingressi_mensili_free["mese_anno"], ordered=True)
#print(ingressi_mensili_free)


# Raggruppo per anno e somma gli ingressi free
ingressi_annuali_free = df.groupby(pd.Grouper(key="datetime", freq="Y"))[["free"]].sum().reset_index()
# Creo colonna con anno (es. '2023')
ingressi_annuali_free["anno"] = ingressi_annuali_free["datetime"].dt.strftime("%Y")
# Ordine cronologico corretto
ingressi_annuali_free["anno"] = pd.Categorical(ingressi_annuali_free["anno"],categories=ingressi_annuali_free["anno"],ordered=True)
#print(ingressi_annuali_free)


# Creazione dataframe per ***ingressi annuali***
ingressi_annuali = pd.merge(
    ingressi_annuali_full_price[["anno", "full_price"]],
    ingressi_annuali_reduced[["anno", "reduced"]],
    on="anno")

# Poi unisco anche free per ***ingressi annuali***
ingressi_annuali = pd.merge(
    ingressi_annuali,
    ingressi_annuali_free[["anno", "free"]],
    on="anno")

#print(ingressi_annuali)


# Creazione dataframe per ***ingressi mensili***
ingressi_mensili = pd.merge(
    ingressi_mensili_full_price[["datetime", "full_price"]],
    ingressi_mensili_reduced[["datetime", "reduced"]],
    on="datetime"
)

# Poi unisco anche free per ***ingressi mensili***
ingressi_mensili = pd.merge(
    ingressi_mensili,
    ingressi_mensili_free[["datetime", "free"]],
    on="datetime"
)


# Creo colonna "mese_anno" (formato leggibile)
ingressi_mensili["mese_anno"] = ingressi_mensili["datetime"].dt.strftime("%b %Y")
ingressi_mensili["mese_anno"] = pd.Categorical(
    ingressi_mensili["mese_anno"],
    categories=ingressi_mensili["mese_anno"],
    ordered=True
)

# Creo colonna "anno" e "mese"
ingressi_mensili["anno"] = ingressi_mensili["datetime"].dt.year
ingressi_mensili["mese"] = ingressi_mensili["datetime"].dt.strftime("%b")

# Ordino per data (utile per i grafici)
ingressi_mensili = ingressi_mensili.sort_values("datetime").reset_index(drop=True)

# Visualizza il risultato
#print(ingressi_annuali.head())
#print(ingressi_mensili.head())

anni = ingressi_annuali["anno"]
full_price = ingressi_annuali["full_price"]
reduced = ingressi_annuali["reduced"]


#### GRAFICI PER EDA ####


# 1) GRAFICO A BARRE IMPILATE Full_price vs Reduced (per ANNO)
plt.bar(anni, full_price, label="Full_price")
plt.bar(anni, reduced, bottom=full_price, label="Reduced")

# Etichette con il totale sopra ogni barra
for i in range(len(anni)):
    totale = full_price.iloc[i] + reduced.iloc[i]
    plt.text(anni[i], totale, f"{int(totale):,}", ha='center', va='bottom', fontsize=9)

#title = "Ingressi per anno: Full_price vs Reduced"
plt.title("Ingressi per anno: Full_price vs Reduced")
plt.xlabel("Anno")
plt.ylabel("Numero ingressi")
plt.legend()
plt.tight_layout()
#safe_title = title.replace(" ", "_").replace("/", "-").lower() + ".png"
plt.savefig(os.path.join(output_dir,"Ingressi_per_anno_Full_price_vs_Reduced.png"), dpi=300)
#plt.show()


# 2) GRAFICO A BARRE per giorno della settimana Full_price vs Reduced (per ANNO)

plt.figure(figsize=(10, 5))
# Ordine dei giorni
ordine_giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]

# Raggruppa per giorno della settimana e ordina
df_giorni = df.groupby("giorno_settimana")[["full_price", "reduced"]].sum().reindex(ordine_giorni)

# Setup grafico
plt.figure(figsize=(10, 5))
x = range(len(df_giorni))

# Barre impilate
plt.bar(x, df_giorni["full_price"], label="Full_price", color="skyblue")
plt.bar(x, df_giorni["reduced"], bottom=df_giorni["full_price"], label="Reduced", color="salmon")

# Etichette numeriche
for i in x:
    full = df_giorni["full_price"].iloc[i]
    red = df_giorni["reduced"].iloc[i]
    tot = full + red

    # Valore full_price
    plt.text(i, full / 2, f"{int(full):,}", ha='center', va='center', fontsize=8, color="black")

    # Valore reduced
    plt.text(i, full + red / 2, f"{int(red):,}", ha='center', va='center', fontsize=8, color="black")

    # Totale (in alto)
    plt.text(i, tot + 5, f"{int(tot):,}", ha='center', va='bottom', fontsize=9, fontweight="bold")

# Layout
plt.xticks(x, ordine_giorni, rotation=45)
plt.xlabel("Giorno della settimana")
plt.ylabel("Totale ingressi")
plt.title("Ingressi per giorno della settimana: Full_price vs Reduced")
plt.legend()
plt.tight_layout()

# Salvataggio
plt.savefig(os.path.join(output_dir,"Ingressi_annuali_per_giorno_della_settimana.png"), dpi=300)
#plt.show()


# 3) GRAFICO A BARRE IMPILATE in base alla stagione Full_price vs Reduced (per ANNO)

# Ordine stagioni (modifica se i tuoi valori sono diversi)
ordine_stagioni = ["primavera", "estate", "autunno", "inverno"]

# Raggruppa per stagione e somma full_price e reduced
df_stagioni = df.groupby("stagione")[["full_price", "reduced"]].sum().reindex(ordine_stagioni)

# Posizioni X
x = range(len(df_stagioni))

# Plot barre impilate
plt.figure(figsize=(8, 5))
plt.bar(x, df_stagioni["full_price"], label="Full_price", color="skyblue")
plt.bar(x, df_stagioni["reduced"], bottom=df_stagioni["full_price"], label="Reduced", color="salmon")

# Etichette numeriche (full, reduced, totale)
for i in x:
    full = df_stagioni["full_price"].iloc[i]
    red = df_stagioni["reduced"].iloc[i]
    tot = full + red

    # Etichetta full_price (centro prima parte)
    plt.text(i, full / 2, f"{int(full):,}", ha='center', va='center', fontsize=8)

    # Etichetta reduced (centro seconda parte)
    plt.text(i, full + red / 2, f"{int(red):,}", ha='center', va='center', fontsize=8)

    # Etichetta totale (sopra barra)
    plt.text(i, tot + 5, f"{int(tot):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Layout
plt.xticks(x, ordine_stagioni)
plt.xlabel("Stagione")
plt.ylabel("Numero ingressi")
plt.title("Ingressi per stagione: Full_price vs Reduced")
plt.legend()
plt.tight_layout()

# Salva il grafico
plt.savefig(os.path.join(output_dir,"Ingressi_per_stagione_full_vs_reduced.png"), dpi=300)
#plt.show()

# 4) GRAFICO A BARRE IMPILATE in base alla fascia oraria Full_price vs Reduced (per ANNO)

# Ordine delle fasce orarie (personalizza se necessario)
ordine_fasce = ["mattina", "pomeriggio", "sera"]

# Raggruppa per fascia oraria e somma ingressi
df_fasce = df.groupby("fascia_oraria")[["full_price", "reduced"]].sum().reindex(ordine_fasce)

# Posizioni X
x = range(len(df_fasce))

# Plot
plt.figure(figsize=(10, 5))
plt.bar(x, df_fasce["full_price"], label="Full_price", color="skyblue")
plt.bar(x, df_fasce["reduced"], bottom=df_fasce["full_price"], label="Reduced", color="salmon")

# Etichette numeriche
for i in x:
    full = df_fasce["full_price"].iloc[i]
    red = df_fasce["reduced"].iloc[i]
    tot = full + red

    # Valore full_price
    plt.text(i, full / 2, f"{int(full):,}", ha='center', va='center', fontsize=8)

    # Valore reduced
    plt.text(i, full + red / 2, f"{int(red):,}", ha='center', va='center', fontsize=8)

    # Totale in cima
    plt.text(i, tot + 5, f"{int(tot):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Layout
plt.xticks(x, ordine_fasce)
plt.xlabel("Fascia oraria")
plt.ylabel("Numero ingressi")
plt.title("Ingressi per fascia oraria: Full_price vs Reduced")
plt.legend()
plt.tight_layout()

# Salvataggio
plt.savefig(os.path.join(output_dir,"Ingressi_per_fascia_oraria_full_vs_reduced.png"), dpi=300)
#plt.show()

# 5) GRAFICO A BARRE IMPILATE Weekend vs Non-Weekend in base alla fascia oraria Full_price vs Reduced (per ANNO)

# Raggruppa per weekend (True/False) e somma ingressi
df_weekend = df.groupby("weekend")[["full_price", "reduced"]].sum()

# Converti booleani in etichette leggibili
df_weekend.index = df_weekend.index.map({False: "Feriale", True: "Weekend"})

# Posizioni X
x = range(len(df_weekend))

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x, df_weekend["full_price"], label="Full_price", color="skyblue")
plt.bar(x, df_weekend["reduced"], bottom=df_weekend["full_price"], label="Reduced", color="salmon")

# Etichette numeriche
for i in x:
    full = df_weekend["full_price"].iloc[i]
    red = df_weekend["reduced"].iloc[i]
    tot = full + red

    plt.text(i, full / 2, f"{int(full):,}", ha='center', va='center', fontsize=8)
    plt.text(i, full + red / 2, f"{int(red):,}", ha='center', va='center', fontsize=8)
    plt.text(i, tot + 5, f"{int(tot):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Layout
plt.xticks(x, df_weekend.index)
plt.xlabel("Tipo di giorno")
plt.ylabel("Numero ingressi")
plt.title("Weekend o Feriale: Full_price vs Reduced")
plt.legend()
plt.tight_layout()

# Salvataggio
plt.savefig(os.path.join(output_dir,"Ingressi_weekend_vs_feriale.png"), dpi=300)
#plt.show()



# 6) GRAFICO ingressi per anno e subplot ingressi per mese Full_price vs Reduced (per ANNO)

mesi = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
df["nome_mese"] = df["mese"].apply(lambda x: mesi[x - 1])
anni = sorted(df["anno"].unique())

# Totali annuali
totali_per_anno = df.groupby("anno")["total"].sum().reset_index()

# Loop sugli anni
for anno_scelto in anni:
    df_anno = df[df["anno"] == anno_scelto]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2 righe, 1 colonna
    sns.set(style="whitegrid")

    # --- GRAFICO SINISTRA: Totale ingressi per anno ---
    sns.barplot(
        data=totali_per_anno,
        x="anno",
        y="total",
        palette="Blues",
        ax=axes[0],
        errorbar=None
    )
    axes[0].set_title("Ingressi totali per anno", fontweight='bold')
    axes[0].set_xlabel("Anno")
    axes[0].set_ylabel("Totale ingressi")

    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{int(height):,}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # --- GRAFICO DESTRA: Full vs Reduced per mese ---
    df_mesi = df_anno.groupby("nome_mese")[["full_price", "reduced"]].sum().reindex(mesi).reset_index()

    df_melt = pd.melt(df_mesi, id_vars="nome_mese", value_vars=["full_price", "reduced"],
                      var_name="Tipo", value_name="Ingressi")

    sns.barplot(
        data=df_melt,
        x="nome_mese",
        y="Ingressi",
        hue="Tipo",
        palette={"full_price": "skyblue", "reduced": "salmon"},
        ax=axes[1],
        errorbar=None
    )

    axes[1].set_title(f"Ingressi mensili - {anno_scelto}", fontweight='bold')
    axes[1].set_xlabel("Mese")
    axes[1].set_ylabel("Numero ingressi")
    axes[1].legend(title="Tipo biglietto")

    # Etichette sopra ogni barra
    for bar in axes[1].patches:
        height = bar.get_height()
        if height > 0:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height + 5,
                f"{int(height):,}",
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"Grafico_ingressi_{anno_scelto}.png"), dpi=300)
    #plt.show()


#### GRAFICI PER CORRELAZIONE ####


# Grafico: Pioggia vs Ingressi Full price
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="precip_mm", y="full_price", alpha=0.6, color="teal")
sns.regplot(data=df, x="precip_mm", y="full_price", scatter=False, color="red")  # Linea di tendenza

plt.title("Relazione tra pioggia (mm) e ingressi Full price")
plt.xlabel("Precipitazioni (mm)")
plt.ylabel("Ingressi totali")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_correlazione_precipitazioni_vs_full_price.png"), dpi=300)
#plt.show()

# Calcolo coefficiente di correlazione Pearson
corr_full, pval_full = stats.pearsonr(df["precip_mm"], df["full_price"])
print(f"[precip_mm vs full_price] Correlazione Pearson: r = {corr_full:.2f}, p = {pval_full:.4f}")


# Grafico: Pioggia vs Ingressi Reduced
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="precip_mm", y="reduced", alpha=0.6, color="teal")
sns.regplot(data=df, x="precip_mm", y="reduced", scatter=False, color="red")  # Linea di tendenza

plt.title("Relazione tra pioggia (mm) e ingressi Reduced")
plt.xlabel("Precipitazioni (mm)")
plt.ylabel("Ingressi totali")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_correlazione_precipitazioni_vs_reduced.png"), dpi=300)
#plt.show()

# Calcolo coefficiente di correlazione Pearson
corr_full, pval_full = stats.pearsonr(df["precip_mm"], df["reduced"])
print(f"[precip_mm vs reduced] Correlazione Pearson: r = {corr_full:.2f}, p = {pval_full:.4f}")


# Grafico: Temperatura media vs Ingressi Full price
plt.figure(figsize=(7, 5))
sns.scatterplot(x=(df["temp_max"] + df["temp_min"]) / 2, y=df["full_price"], alpha=0.6, color="teal")
sns.regplot(x=(df["temp_max"] + df["temp_min"]) / 2, y=df["total"], scatter=False, color="red")

plt.title("Relazione tra temperatura media e ingressi Full price")
plt.xlabel("Temperatura media (°C)")
plt.ylabel("Ingressi Full price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_correlazione_temp_media_vs_full_price.png"), dpi=300)
#plt.show()

# Calcolo coefficiente di correlazione Pearson
temp_media = (df["temp_max"] + df["temp_min"]) / 2
r, p = stats.pearsonr(temp_media, df["full_price"])
print(f"[temp_media vs full_price] Correlazione Pearson: r = {r:.2f}, p = {p:.4f}")

# Grafico: Temperatura media vs Ingressi Reduced
plt.figure(figsize=(7, 5))
sns.scatterplot(x=(df["temp_max"] + df["temp_min"]) / 2, y=df["reduced"], alpha=0.6, color="teal")
sns.regplot(x=(df["temp_max"] + df["temp_min"]) / 2, y=df["reduced"], scatter=False, color="red")

plt.title("Relazione tra temperatura media e ingressi Reduced")
plt.xlabel("Temperatura media (°C)")
plt.ylabel("Ingressi reduced")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_correlazione_temp_media_vs_reduced.png"), dpi=300)
#plt.show()

# Calcolo coefficiente di correlazione Pearson
temp_media = (df["temp_max"] + df["temp_min"]) / 2
r, p = stats.pearsonr(temp_media, df["reduced"])
print(f"[temp_media vs reduced] Correlazione Pearson: r = {r:.2f}, p = {p:.4f}")


# Stagione (ANOVA) - Full_price -

ordine_stagioni = ["inverno", "primavera", "estate", "autunno"]

plt.figure(figsize=(8, 5))
ordine_stagioni = ["inverno", "primavera", "estate", "autunno"]
sns.boxplot(data=df, x="stagione", y="full_price", order=ordine_stagioni, hue="stagione", palette="Set2")
plt.title("Distribuzione degli ingressi Full_price per stagione")
plt.xlabel("Stagione")
plt.ylabel("Ingressi Full price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_ANOVA_stagione_vs_full_price.png"), dpi=300)
#plt.show()

gruppi = [df[df["stagione"] == s]["full_price"] for s in ordine_stagioni]
stat_s, p_s = f_oneway(*gruppi)
print(f"[stagione vs full_price] ANOVA → F = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test ANOVA è significativo: ci sono differenze statisticamente rilevanti tra le stagioni.")
else:
    print("→ Il test ANOVA non è significativo: non ci sono differenze significative tra le stagioni.")


# Stagione (ANOVA) - Reduced -

plt.figure(figsize=(8, 5))
ordine_stagioni = ["inverno", "primavera", "estate", "autunno"]
sns.boxplot(data=df, x="stagione", y="reduced", order=ordine_stagioni, hue="stagione", palette="Set2")
plt.title("Distribuzione degli ingressi Reduced per stagione")
plt.xlabel("Stagione")
plt.ylabel("Ingressi Reduced")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_ANOVA_stagione_vs_reduced.png"), dpi=300)
#plt.show()

gruppi = [df[df["stagione"] == s]["reduced"] for s in ordine_stagioni]
stat_s, p_s = f_oneway(*gruppi)
print(f"[stagione vs reduced] ANOVA → F = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test ANOVA è significativo: ci sono differenze statisticamente rilevanti tra le stagioni.")
else:
    print("→ Il test ANOVA non è significativo: non ci sono differenze significative tra le stagioni.")



# Fascia oraria (T-test) - Full price -

# Filtra il DataFrame escludendo la fascia "mattina"
df_fasce = df[df["fascia_oraria"].isin(["pomeriggio", "sera"])]

# Boxplot aggiornato
plt.figure(figsize=(8, 5))
ordine_fasce = ["pomeriggio", "sera"]
sns.boxplot(data=df_fasce, x="fascia_oraria", y="full_price", order=ordine_fasce, hue="fascia_oraria", palette="Set2")
plt.title("Distribuzione degli ingressi Full price per fascia oraria (mattina esclusa)")
plt.xlabel("Fascia oraria")
plt.ylabel("Ingressi Full price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_TTEST_fascia_oraria_vs_full_price.png"), dpi=300)
#plt.show()

# T-test tra pomeriggio e sera
gruppo1 = df_fasce[df_fasce["fascia_oraria"] == "pomeriggio"]["full_price"]
gruppo2 = df_fasce[df_fasce["fascia_oraria"] == "sera"]["full_price"]
stat_s, p_s = ttest_ind(gruppo1, gruppo2, equal_var=False)

print(f"[fascia_oraria vs full_price] T-test → t = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test è significativo: ci sono differenze statisticamente rilevanti tra pomeriggio e sera.")
else:
    print("→ Il test non è significativo: non ci sono differenze significative tra pomeriggio e sera.")


# Fascia oraria (T-test) - Reduced -

# Filtra il DataFrame escludendo la fascia "mattina"
df_fasce = df[df["fascia_oraria"].isin(["pomeriggio", "sera"])]

# Boxplot aggiornato
plt.figure(figsize=(8, 5))
ordine_fasce = ["pomeriggio", "sera"]
sns.boxplot(data=df_fasce, x="fascia_oraria", y="reduced", order=ordine_fasce, hue="fascia_oraria", palette="Set2")
plt.title("Distribuzione degli ingressi Reduced per fascia oraria (mattina esclusa)")
plt.xlabel("Fascia oraria")
plt.ylabel("Ingressi Reduced")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_TTEST_fascia_oraria_vs_reduced.png"), dpi=300)
#plt.show()

# T-test tra pomeriggio e sera
gruppo1 = df_fasce[df_fasce["fascia_oraria"] == "pomeriggio"]["reduced"]
gruppo2 = df_fasce[df_fasce["fascia_oraria"] == "sera"]["reduced"]
stat_s, p_s = ttest_ind(gruppo1, gruppo2, equal_var=False)

print(f"[fascia_oraria vs reduced] T-test → t = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test è significativo: ci sono differenze statisticamente rilevanti tra pomeriggio e sera.")
else:
    print("→ Il test non è significativo: non ci sono differenze significative tra pomeriggio e sera.")


# Giorno della settimana (ANOVA) - Full price -

# Boxplot
plt.figure(figsize=(10, 5))
ordine_giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
sns.boxplot(data=df, x="giorno_settimana", y="full_price", order=ordine_giorni, palette="Set2")
plt.title("Distribuzione degli ingressi Full price per giorno della settimana")
plt.xlabel("Giorno della settimana")
plt.ylabel("Ingressi Full price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_ANOVA_giorno_settimana_vs_full_price.png"), dpi=300)
#plt.show()

# ANOVA test
gruppi = [df[df["giorno_settimana"] == g]["full_price"] for g in ordine_giorni]
stat_s, p_s = f_oneway(*gruppi)
print(f"[giorno_settimana vs full_price] ANOVA → F = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test ANOVA è significativo: ci sono differenze statisticamente rilevanti tra i giorni della settimana.")
else:
    print("→ Il test ANOVA non è significativo: non ci sono differenze significative tra i giorni della settimana.")


# Giorno della settimana (ANOVA) - Reduced -

# Boxplot
plt.figure(figsize=(10, 5))
ordine_giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
sns.boxplot(data=df, x="giorno_settimana", y="reduced", order=ordine_giorni, palette="Set2")
plt.title("Distribuzione degli ingressi Reduced per giorno della settimana")
plt.xlabel("Giorno della settimana")
plt.ylabel("Ingressi Reduced")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_ANOVA_giorno_settimana_vs_reduced.png"), dpi=300)
#plt.show()

# ANOVA test
gruppi = [df[df["giorno_settimana"] == g]["reduced"] for g in ordine_giorni]
stat_s, p_s = f_oneway(*gruppi)
print(f"[giorno_settimana vs reduced] ANOVA → F = {stat_s:.2f}, p = {p_s:.4e}")
if p_s < 0.05:
    print("→ Il test ANOVA è significativo: ci sono differenze statisticamente rilevanti tra i giorni della settimana.")
else:
    print("→ Il test ANOVA non è significativo: non ci sono differenze significative tra i giorni della settimana.")


# Weekend (t-test) - Full price -
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="weekend", y="full_price", hue="weekend", palette="coolwarm")
plt.title("Distribuzione degli ingressi Full price: Weekend vs Giorni Feriali")
plt.xlabel("Weekend")
plt.ylabel("Ingressi Full price")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_TTEST_weekend_o_no_vs_full_price.png"), dpi=300)
#plt.show()

# T-test
gruppo_weekend = df[df["weekend"] == True]["full_price"]
gruppo_feriali = df[df["weekend"] == False]["full_price"]
stat_t, p_t = ttest_ind(gruppo_weekend, gruppo_feriali, equal_var=False)
print(f"[weekend vs full_price] t-test → t = {stat_t:.2f}, p = {p_t:.4e}")

if p_t < 0.05:
    print("→ Il test è significativo: ci sono differenze statisticamente rilevanti tra weekend e non-weekend.")
else:
    print("→ Il test non è significativo: non ci sono differenze significative tra weekend e non-weekend.")

# Weekend (t-test) - Reduced -
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="weekend", y="reduced", hue="weekend", palette="coolwarm")
plt.title("Distribuzione degli ingressi Reduced: Weekend vs Giorni Feriali")
plt.xlabel("Weekend")
plt.ylabel("Ingressi Reduced")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_TTEST_weekend_o_no_vs_reduced.png"), dpi=300)
#plt.show()

# T-test
gruppo_weekend = df[df["weekend"] == True]["reduced"]
gruppo_feriali = df[df["weekend"] == False]["reduced"]
stat_t, p_t = ttest_ind(gruppo_weekend, gruppo_feriali, equal_var=False)
print(f"[weekend vs reduced] t-test → t = {stat_t:.2f}, p = {p_t:.4e}")

if p_t < 0.05:
    print("→ Il test è significativo: ci sono differenze statisticamente rilevanti tra weekend e non-weekend.")
else:
    print("→ Il test non è significativo: non ci sono differenze significative tra weekend e non-weekend.")



#### TUKEY HSD

# Tukey HSD per stagione - Full price -
print("\n>>> Post-hoc: Tukey HSD per 'stagione' (full_price)")
tukey_stagione = pairwise_tukeyhsd(endog=df["full_price"], groups=df["stagione"], alpha=0.05)
print(tukey_stagione.summary())

# Grafico
tukey_stagione.plot_simultaneous()
plt.title("Tukey HSD – Ingressi Full Price per Stagione")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_Tukey_HSD_stagione_vs_full_price.png"), dpi=300)
#plt.show()

# Tukey HSD per stagione - Reduced -
print("\n>>> Post-hoc: Tukey HSD per 'stagione' (reduced)")
tukey_stagione = pairwise_tukeyhsd(endog=df["reduced"], groups=df["stagione"], alpha=0.05)
print(tukey_stagione.summary())

# Grafico
tukey_stagione.plot_simultaneous()
plt.title("Tukey HSD – Ingressi Reduced per Stagione")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"Grafico_Tukey_HSD_stagione_vs_reduced.png"), dpi=300)
#plt.show()
