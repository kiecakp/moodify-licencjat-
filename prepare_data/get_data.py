import os
import csv
import time
import pandas as pd
import lyricsgenius
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload, TranslationNotFound
from dotenv import load_dotenv
from threading import Lock

load_dotenv()

INPUT_XLSX  = "../data/ml_balanced.xlsx"
OUTPUT_CSV  = "../data/training_data.csv"
SAVE_EVERY  = 20
WORKERS     = 5   # liczba rownoleglych watkow (nie przekraczaj 8 - limit Genius)

# -------- Genius API (osobna instancja na watek) ----------------------------

GENIUS_TOKEN = os.getenv("GENIUS_API_TOKEN")

def make_genius():
    g = lyricsgenius.Genius(GENIUS_TOKEN)
    g.skip_non_songs = True
    g.excluded_terms = ["(Remix)", "(Live)"]
    g.remove_section_headers = True
    g.verbose = False
    g.timeout = 8
    g.retries = 1
    return g

# thread-local storage - kazdy watek ma swoja instancje Genius
import threading
_local = threading.local()

def get_genius():
    if not hasattr(_local, "genius"):
        _local.genius = make_genius()
    return _local.genius

# -------- Wczytywanie danych ----------------------------

def load_from_xlsx(path: str) -> list[dict]:
    df = pd.read_excel(path, header=None)

    header_row = next(
        (i for i, row in df.iterrows() if str(row[0]).strip() == "Index"), None
    )

    if header_row is None:
        raise ValueError(f"Nie znaleziono naglowka 'Index' w: {path}")

    data = df.iloc[header_row + 1:].copy()
    data.columns = ["index", "artist", "title", "mood"]
    data = data.dropna(subset=["artist", "title", "mood"]).reset_index(drop=True)
    print(f"  {path}: {len(data)} piosenek")
    return data.to_dict(orient="records")

# -------- Obsluga pliku wyjsciowego ----------------------------

def ensure_csv_exists():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["TITLE", "ARTIST", "MOOD_TAG", "LYRICS", "LYRICS_LANGUAGE", "LYRICS_EN"])
        print("Utworzono nowy plik training_data.csv")
    else:
        print("Plik training_data.csv juz istnieje - dopisywanie danych\n")

def get_existing_tracks() -> set:
    if not os.path.exists(OUTPUT_CSV):
        return set()

    with open(OUTPUT_CSV, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return {(row[0].strip().lower(), row[1].strip().lower()) for row in reader if len(row) >= 2}

# -------- Przetwarzanie jednej piosenki ----------------------------

def detect_language(text: str) -> str | None:
    try:
        return detect(text) if text and text.strip() else None
    except Exception:
        return None

def translate_to_english(text: str, lang: str) -> str | None:
    try:
        if lang == "en":
            return text
        translated = GoogleTranslator(source=lang, target="en").translate(text)
        return translated if translated and translated.strip() else None
    except Exception:
        return None

def process_one(song: dict) -> tuple[str, list | None]:
    """
    Zwraca (status, row) gdzie status to:
    'ok', 'no_lyrics', 'lang_fail', 'trans_fail'
    """
    title  = str(song["title"]).strip()
    artist = str(song["artist"]).strip()
    mood   = str(song["mood"]).strip()

    try:
        g = get_genius()
        result = g.search_song(title, artist)
        lyrics = result.lyrics if result and result.lyrics else ""
    except Exception as e:
        lyrics = ""

    if not lyrics.strip():
        return ("no_lyrics", None)

    lang = detect_language(lyrics)
    if not lang:
        return ("lang_fail", None)

    if lang != "en":
        lyrics_en = translate_to_english(lyrics, lang)
        if not lyrics_en:
            return ("trans_fail", None)
    else:
        lyrics_en = lyrics

    return ("ok", [title, artist, mood, lyrics, lang, lyrics_en])

# -------- Glowna petla z watkami ----------------------------

def process_songs(songs: list[dict]):
    existing = get_existing_tracks()
    
    # Filtruj duplikaty z gory
    to_process = [
        s for s in songs
        if (str(s["title"]).strip().lower(), str(s["artist"]).strip().lower()) not in existing
    ]
    skipped = len(songs) - len(to_process)
    total   = len(to_process)

    print(f"Do pobrania: {total} piosenek | Pominiete duplikaty: {skipped}")
    print(f"Watki: {WORKERS} | Szacowany czas: ~{total * 3 // WORKERS // 60} minut\n")

    stats  = {"ok": 0, "no_lyrics": 0, "lang_fail": 0, "trans_fail": 0}
    buffer = []
    lock   = Lock()
    done   = 0

    def save_buffer():
        nonlocal buffer
        if buffer:
            with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(buffer)
            buffer = []

    start = time.time()

    # tworzymy pule 5 watkow
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_one, song): song for song in to_process}

        for future in as_completed(futures):
            song   = futures[future]
            title  = str(song["title"]).strip()
            artist = str(song["artist"]).strip()

            try:
                status, row = future.result()
            except Exception as e:
                status, row = "no_lyrics", None

            with lock:
                done += 1
                stats[status] = stats.get(status, 0) + 1

                if status == "ok" and row:
                    buffer.append(row)

                # Podsumowanie co SAVE_EVERY
                if done % SAVE_EVERY == 0 or done == total:
                    save_buffer()
                    elapsed  = time.time() - start
                    per_song = elapsed / done
                    remaining = int((total - done) * per_song)
                    print(
                        f"[{done}/{total}] "
                        f"ok={stats['ok']} brak={stats['no_lyrics']} "
                        f"| pozostalo ~{remaining//60}m {remaining%60}s"
                    )

    elapsed = int(time.time() - start)
    print(f"""
        Gotowe! Czas: {elapsed//60}m {elapsed%60}s
        Pobrano tekstow:        {stats['ok']}
        Brak tekstu:            {stats['no_lyrics']}
        Blad jezyka:            {stats.get('lang_fail', 0)}
        Blad tlumaczenia:       {stats.get('trans_fail', 0)}
        Pominiete (duplikaty):  {skipped}
        Wynik zapisany w:       {OUTPUT_CSV}
    """)

if __name__ == "__main__":
    ensure_csv_exists()
    songs = load_from_xlsx(INPUT_XLSX)
    process_songs(songs)