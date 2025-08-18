import pandas as pd
import locale
from holidays import country_holidays
import matplotlib.pyplot as plt
import seaborn as sns
import requests
#from tmdbv3api import TMDb, Movie
import os
from dotenv import load_dotenv

load_dotenv()

# Stile dei grafici
sns.set(style="whitegrid")

# Mappa dei giorni in italiano (corretta e sicura)
giorni_settimana = {
    "Monday": "lunedì",
    "Tuesday": "martedì",
    "Wednesday": "mercoledì",
    "Thursday": "giovedì",
    "Friday": "venerdì",
    "Saturday": "sabato",
    "Sunday": "domenica"
}

# Percorso assoluto basato sulla posizione del file corrente
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "input", "dati_cinema_originali.csv")


# Caricamento con encoding
df = pd.read_csv(path, encoding="latin1", sep=";", dayfirst=True, on_bad_lines="skip")
print(df.head())

# Conversione della data
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

# Combina data + ora reale
df['datetime'] = pd.to_datetime(df['date'].dt.date.astype(str) + ' ' + df['time'], errors='coerce')

# Ordina per data
df = df.sort_values("datetime").reset_index(drop=True)


# Estrazione e conversione
df["giorno_settimana_en"] = df["datetime"].dt.day_name()
df["giorno_settimana"] = df["giorno_settimana_en"].map(giorni_settimana)


# Pulizia colonna intermedia
df.drop(columns=["giorno_settimana_en"], inplace=True)


# Mese, anno, ora
df["mese"] = df["datetime"].dt.month
df["anno"] = df["datetime"].dt.year
df["ora"] = df["datetime"].dt.hour


# Fascia oraria
def get_fascia_oraria(ora):
    if ora == 10:
        return "mattina"
    elif 15 <= ora <= 17:
        return "pomeriggio"
    elif 20 <= ora <= 22:
        return "sera"
    else:
        return "altro"  # in caso ci fossero altri orari imprevisti

df["fascia_oraria"] = df["ora"].apply(get_fascia_oraria)


# Funzione per classificare le stagioni
def get_stagione(mese):
    if mese in [12, 1, 2]:
        return "inverno"
    elif mese in [3, 4, 5]:
        return "primavera"
    elif mese in [6, 7, 8]:
        return "estate"
    else:
        return "autunno"

df["stagione"] = df["mese"].apply(get_stagione)

# Weekend: True se sabato o domenica
df["weekend"] = df["giorno_settimana"].isin(["venerdì", "sabato", "domenica"])

# Anni unici presenti nel dataset
anni_presenti = df["anno"].unique()

# Festività italiane ufficiali per quegli anni
festivita_italiane = country_holidays("IT", years=anni_presenti)

# Colonna con nome della festività (NaN se non è una festività)
df["festività"] = df["datetime"].dt.date.map(festivita_italiane)
# Sostituisce i valori NaN con 'Nessuna'
df["festività"] = df["festività"].fillna("Nessuna")


df['total'] = df[['full_price', 'reduced', 'free']].sum(axis=1)
df['full_price'] = df['full_price'].fillna(0)
df['reduced'] = df['reduced'].fillna(0)
df['free'] = df['free'].fillna(0)


#Creazione file
#df.to_excel("C:\\Users\\deidd\\Downloads\\dati_cinema_new.xlsx", index=False)


#### INTEGRAZIONE DATI METEO ####

def get_meteo(date, lat=46.0524, lon=11.45):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=Europe%2FRome"
    )
    response = requests.get(url)
    data = response.json()
    if "daily" in data:
        return {
            "temp_max": data["daily"]["temperature_2m_max"][0],
            "temp_min": data["daily"]["temperature_2m_min"][0],
            "precip_mm": data["daily"]["precipitation_sum"][0]
        }
    else:
        return {"temp_max": None, "temp_min": None, "precip": None}

df["data_str"] = df["datetime"].dt.date.astype(str)
meteo = df["data_str"].apply(get_meteo)
meteo_df = pd.DataFrame(meteo.tolist())

# Unisci al dataframe originale
df = pd.concat([df, meteo_df], axis=1)


#### INTEGRAZIONE DATI TMDB ####

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def get_tmdb_info(title, year=None, api_key=TMDB_API_KEY):
    query = title.replace(" ", "+")
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"

    try:
        res = requests.get(url)
        res.raise_for_status()
        results = res.json().get("results", [])
    except Exception as e:
        print(f"[✘] Errore nella richiesta per '{title}': {e}")
        return {
            "genres": None,
            "keywords": None,
            "cast": None,
            "director": None,
            "vote_average": None,
            "popularity": None
        }

    if not results:
        print(f"[✘] Titolo NON trovato: '{title}'")
        return {
            "genres": None,
            "keywords": None,
            "cast": None,
            "director": None,
            "vote_average": None,
            "popularity": None
        }

    # Se c'è un solo risultato, usalo direttamente
    if len(results) == 1:
        movie = results[0]
    else:
        # Più di un risultato → usa l'anno per scegliere il più vicino
        matches = [r for r in results if r.get("release_date", "").startswith(str(year))] if year else []
        movie = matches[0] if matches else results[0]

    movie_id = movie["id"]
    print(f"[✓] Trovato: '{movie['title']}' → ID {movie_id} (anno: {movie.get('release_date')})")

    detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits,keywords"

    try:
        d = requests.get(detail_url).json()
    except Exception as e:
        print(f"[✘] Errore nella richiesta dettagliata per ID {movie_id}: {e}")
        return {
            "genres": None,
            "keywords": None,
            "cast": None,
            "director": None,
            "vote_average": None,
            "popularity": None
        }

    genres = [g["name"] for g in d.get("genres", [])]
    keywords = [k["name"] for k in d.get("keywords", {}).get("keywords", [])]
    cast = [c["name"] for c in d.get("credits", {}).get("cast", [])[:3]]
    director = next((c["name"] for c in d.get("credits", {}).get("crew", []) if c["job"] == "Director"), None)

    return {
        "genres": genres,
        "keywords": keywords,
        "cast": cast,
        "director": director,
        "vote_average": d.get("vote_average"),
        "popularity": d.get("popularity")
    }

tmdb_info = df.apply(lambda row: get_tmdb_info(row["title"], row["anno"]), axis=1)
tmdb_df = pd.DataFrame(tmdb_info.tolist())
df = pd.concat([df.reset_index(drop=True), tmdb_df.reset_index(drop=True)], axis=1)

print(df)

# Usa la stessa directory di input calcolata all'inizio
input_dir = os.path.join(script_dir, "input")

