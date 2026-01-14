import os
import csv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import lyricsgenius
from langdetect import detect
from deep_translator import GoogleTranslator

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

  # Save to CSV
  with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for item in tracks:
      track = item.get("track")
      if not track:
        continue
      
      title = track.get("name", "")
      artist = track["artists"][0]["name"] if track.get("artists") else ""
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

  updated_rows = []
  total = len(rows)
  changes_since_save = 0
  skipped_count = 0
  removed_count = 0
  
  for idx, row in enumerate(rows, 1):
    title = row[0]
    artist = row[1]
    
    # Check if lyrics already exist
    if row[6].strip():
      print(f"[{idx}/{total}] ⏭️  Skipping {title} - already has lyrics")
      updated_rows.append(row)
      skipped_count += 1
      continue
    
    print(f"[{idx}/{total}] 🔍 Fetching: {title} by {artist}")
    lyrics = fetch_lyrics_for_track(title, artist)

    if not lyrics.strip():
      print(f"[{idx}/{total}] ❌ No lyrics found - REMOVING from dataset")
      removed_count += 1
      continue  # Skip adding this row

    row[6] = lyrics
    row[7] = detect_language(lyrics) if lyrics else ""
    row[8] = translate_to_english(lyrics) if row[7] != "en" else lyrics
    updated_rows.append(row)
    changes_since_save += 1
    
    # Save progress periodically
    if changes_since_save >= save_every or idx == total:
      print(f"💾 Saving progress... ({len(updated_rows)}/{total} tracks kept)")
      with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)
      changes_since_save = 0

  print(f"\n✅ Processing complete!")
  print(f"   📊 Total processed: {total}")
  print(f"   ⏭️  Already had lyrics: {skipped_count}")
  print(f"   ✅ Successfully fetched: {len(updated_rows) - skipped_count}")
  print(f"   ❌ Removed (no lyrics): {removed_count}")
  print(f"   📁 Final dataset size: {len(updated_rows)} tracks\n")

def detect_language(text):
  try:
    return detect(text)
  except:
    return "unknown"

def translate_to_english(text):
  try:
    # Detect text language
    detected_lang = detect_language(text)
        
    # If text is already in English, return it unchanged
    if detected_lang == 'en':
      return text
        
    # Translate to English
    translator = GoogleTranslator(source=detected_lang, target='en')
    translated = translator.translate(text)
        
    return translated
        
  except Exception as e:
    print(f"❌ Error translating text: {e}")
    return None

if __name__ == "__main__":
  playlists = {
    "happy": [
      "https://open.spotify.com/playlist/6HJJTrh5NWDr1sPhlvsprC"
    ],
    "chill": [
      "https://open.spotify.com/playlist/1Q8dkQapMQEtvjSpwX3TF5",
      "https://open.spotify.com/playlist/4YPmYcv3ApsLjYtnrmxBlj"
    ],
    "energetic": [
      "https://open.spotify.com/playlist/4FRszV25nNzLh049e40XT4",
      "https://open.spotify.com/playlist/1aDiFEvcaGkCX6XuhzpCge"
    ],
    "sad": [
      "https://open.spotify.com/playlist/1iJ8iy1XIFDppTC8ZNlPiz",
      "https://open.spotify.com/playlist/4KXWa5lSs3Mfc624iblABa"
    ]
  }

  ensure_csv_exists()
  
  for mood_tag, playlist_urls in playlists.items():
    for url in playlist_urls:
      fetch_tracks_and_save(url, mood_tag)

  fetch_lyrics_for_all_tracks()