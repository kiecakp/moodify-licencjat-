import os
import csv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import lyricsgenius
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload, TranslationNotFound

load_dotenv()
GENIUS_TOKEN = os.getenv("GENIUS_API_TOKEN")

genius = lyricsgenius.Genius(GENIUS_TOKEN)
genius.skip_non_songs = True 
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.remove_section_headers = True
genius.verbose = False
genius.timeout = 10

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="playlist-read-private playlist-read-collaborative"
))

CSV_FILE = "ml/data/training_data.csv"

def ensure_csv_exists():
  # Create the CSV file with headers if it doesn't exist
  if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TITLE",
            "ARTIST",
            "ALBUM",
            "RELEASE_DATE",
            "MOOD_TAG",
            "EXPICIT",
            "LYRICS",
            "LYRICS_LANGUAGE",
            "LYRICS_EN"
        ])
    print("📄 A new training_data.csv file has been created")
  else:
    print("📄 The training_data.csv file already exists - adding data")

def get_existing_tracks():
  existing = set()
  
  if not os.path.exists(CSV_FILE):
    return existing
  
  with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
      if len(row) >= 2:
        # Create unique identifier from title and artist
        title = row[0].strip().lower()
        artist = row[1].strip().lower()
        existing.add((title, artist))
  
  return existing

def fetch_tracks_and_save(playlist_url, mood_tag):
  playlist_id = playlist_url.split("/")[-1].split("?")[0]

  print("\n🎵 Statting to fetch tracks...")

  # Fetch the first batch
  results = sp.playlist_items(playlist_id, limit=100, market="PL")
  tracks = results["items"]

  # Pagination
  while results["next"]:
      results = sp.next(results)
      tracks.extend(results["items"])

  print(f"\n📚 Found {len(tracks)} tracks.")

  # Get existing tracks to avoid duplicates
  existing_tracks = get_existing_tracks()
  
  new_tracks = 0
  skipped_duplicates = 0

  # Save to CSV
  with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for item in tracks:
      track = item.get("track")
      if not track:
        continue
      
      title = track.get("name", "")
      artist = track["artists"][0]["name"] if track.get("artists") else ""

      # Check if track already exists
      track_id = (title.strip().lower(), artist.strip().lower())
      if track_id in existing_tracks:
        print(f"⏭️  Skipping duplicate: {title} by {artist}")
        skipped_duplicates += 1
        continue

      album = track["album"]["name"] if track.get("album") else ""
      release_date = track["album"].get("release_date", "") if track.get("album") else ""
      mood_tag_value = mood_tag
      explicit = track.get("explicit", False)

      writer.writerow([
          title,
          artist,
          album,
          release_date,
          mood_tag_value,
          explicit,
          "",  # Placeholder for lyrics
          "",  # Placeholder for text language
          ""   # Placeholder for text_EN
      ])

      existing_tracks.add(track_id)
      new_tracks += 1
      print(f"+ Added: {title} by {artist} to existing tracks.")

  print(f"   ⏭️  Duplicates skipped: {skipped_duplicates}\n")
  print("\n✅ All data has been saved to training_data.csv\n")

def fetch_lyrics_for_track(title, artist):
  try:
    song = genius.search_song(title, artist)
    if song and song.lyrics:
      return song.lyrics
    else:
      return ""

  except Exception as e:
    print(f"❌ Error fetching lyrics for {title} by {artist}: {e}")
    return ""

def fetch_lyrics_for_all_tracks(save_every=10):
  with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    headers = reader[0]
    rows = reader[1:]

  updated_rows = []  # Zaczynamy od pustej listy!
  total = len(rows)
  changes_since_save = 0
  skipped_count = 0
  removed_no_lyrics = 0
  removed_translation_failed = 0
  
  for idx, row in enumerate(rows, 1):
    title = row[0]
    artist = row[1]
    
    # Check if lyrics already exist
    if row[6].strip() and row[8].strip():
      print(f"[{idx}/{total}] ⏭️  Skipping {title} - already processed")
      updated_rows.append(row)  # Dodajemy już przetworzone utwory
      skipped_count += 1
      continue
    
    print(f"[{idx}/{total}] 🔍 Fetching: {title} by {artist}")

    # Fetch lyrics if not present
    if not row[6].strip():
      lyrics = fetch_lyrics_for_track(title, artist)
      
      if not lyrics.strip():
        print(f"[{idx}/{total}] ❌ No lyrics found - REMOVING from dataset")
        removed_no_lyrics += 1
        continue  # NIE dodajemy do updated_rows - utwór zostanie usunięty
      
      row[6] = lyrics
    else:
      lyrics = row[6]
    
    # Detect language
    detected_lang = detect_language(lyrics)
    if not detected_lang:
      print(f"[{idx}/{total}] ❌ Language detection failed - REMOVING from dataset")
      removed_translation_failed += 1
      continue  # NIE dodajemy do updated_rows
    
    row[7] = detected_lang
    
    # Translate to English if needed
    if detected_lang != "en":
      print(f"[{idx}/{total}] 🌐 Translating from {detected_lang} to English...")
      translated = translate_to_english(lyrics, detected_lang)
      
      if not translated:
        print(f"[{idx}/{total}] ❌ Translation failed - REMOVING from dataset")
        removed_translation_failed += 1
        continue  # NIE dodajemy do updated_rows
      
      row[8] = translated
    else:
      row[8] = lyrics
    
    # TYLKO tutaj dodajemy utwór do updated_rows (jeśli wszystko się powiodło)
    updated_rows.append(row)
    changes_since_save += 1
    
    # Save progress periodically
    if changes_since_save >= save_every or idx == total:
      print(f"💾 Saving progress... ({len(updated_rows)}/{idx} tracks kept)")
      with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)
      changes_since_save = 0

  print(f"\n✅ Processing complete!")
  print(f"   📊 Total processed: {total}")
  print(f"   ⏭️  Already had data: {skipped_count}")
  print(f"   ✅ Successfully processed: {len(updated_rows) - skipped_count}")
  print(f"   ❌ Removed (no lyrics): {removed_no_lyrics}")
  print(f"   ❌ Removed (translation failed): {removed_translation_failed}")
  print(f"   📁 Final dataset size: {len(updated_rows)} tracks\n")

def detect_language(text):
  try:
    if not text or not text.strip():
      return None
    return detect(text)
  except LangDetectException as e:
    print(f"⚠️  Language detection failed: {e}")
    return None
  except Exception as e:
    print(f"⚠️  Unexpected error in language detection: {e}")
    return None

def translate_to_english(text, detected_lang):
  #Translate text to English. Returns None if translation fails
  try:
    # If text is empty or None
    if not text or not text.strip():
      print(f"⚠️  Empty text, cannot translate")
      return None
    
    # If language detection failed
    if not detected_lang or detected_lang == "unknown":
      print(f"⚠️  Unknown language, cannot translate")
      return None
    
    # If text is already in English, return it unchanged
    if detected_lang == 'en':
      return text
    
    # Try to translate to English
    translator = GoogleTranslator(source=detected_lang, target='en')
    translated = translator.translate(text)
    
    # Check if translation was successful
    if not translated or not translated.strip():
      print(f"⚠️  Translation returned empty result")
      return None
    
    return translated
    
  except (TranslationNotFound, NotValidPayload) as e:
    print(f"⚠️  Translation error: {e}")
    return None
  except Exception as e:
    print(f"⚠️  Unexpected translation error: {e}")
    return None

if __name__ == "__main__":
  playlists = {
    "happy": [
      "https://open.spotify.com/playlist/6HJJTrh5NWDr1sPhlvsprC"
    ],
    "chill": [
      "https://open.spotify.com/playlist/1Q8dkQapMQEtvjSpwX3TF5",
      "https://open.spotify.com/playlist/4YPmYcv3ApsLjYtnrmxBlj",
      "https://open.spotify.com/playlist/1kP7p1pYQ8SBixphVEpTWI?si=iKaY6-VAStezefZnQE1OFw",
      "https://open.spotify.com/playlist/0k0WKMaoZs46MFTYHwZku5?si=ybZxCYJBTrex9jFm3IjDAA"
    ],
    "energetic": [
      "https://open.spotify.com/playlist/4FRszV25nNzLh049e40XT4",
      "https://open.spotify.com/playlist/1aDiFEvcaGkCX6XuhzpCge",
      "https://open.spotify.com/playlist/3nzFGJRv4P7avktrDg8uro?si=e03Am_1CSo6O_B-NIxJw_w",
      "https://open.spotify.com/playlist/3jbslfKMLLri91JYUXqEKz?si=LqYAwx8TS8K6hlSbszi42g"
    ],
    "sad": [
      "https://open.spotify.com/playlist/1iJ8iy1XIFDppTC8ZNlPiz",
      "https://open.spotify.com/playlist/4KXWa5lSs3Mfc624iblABa",
      "https://open.spotify.com/playlist/5mLIo3cu4cJRcdjzfMpQZt?si=5zts0u4aQ-WvH-tSME6r3g",
      "https://open.spotify.com/playlist/3c0Nv5CY6TIaRszlTZbUFk?si=yMuBA4ALRmq76ufO79RESA",
      "https://open.spotify.com/playlist/2ZnAWYy4AOs8tpRUCGF6Py?si=Um6SybesRGmVkE7S6KOGiw",
      "https://open.spotify.com/playlist/4XAU8RJylpzxltN8R7vfzZ?si=O3KUhfXQTuuCYbHHe6twvA"
    ]
  }

  ensure_csv_exists()
  
  for mood_tag, playlist_urls in playlists.items():
    for url in playlist_urls:
      fetch_tracks_and_save(url, mood_tag)

  fetch_lyrics_for_all_tracks()