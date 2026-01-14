import os
import sys
import django
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import lyricsgenius
from langdetect import detect
from deep_translator import GoogleTranslator
from datetime import datetime

# Django setup - adjust the path to project structure
current_dir = os.path.dirname(os.path.abspath(__file__))  # music_data
database_dir = os.path.dirname(current_dir)  # database (zewnętrzny)
backend_dir = os.path.dirname(database_dir)  # backend

sys.path.insert(0, backend_dir)

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "database.database.settings")
django.setup()

from database.models import Track

# Env
load_dotenv()

# Spotify API
sp = spotipy.Spotify(
  auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="playlist-read-private playlist-read-collaborative",
  )
)

# Genius API
GENIUS_TOKEN = os.getenv("GENIUS_API_TOKEN")
genius = lyricsgenius.Genius(GENIUS_TOKEN)
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.remove_section_headers = True
genius.verbose = False
genius.timeout = 10

def fetch_lyrics(title, artist):
  try:
    song = genius.search_song(title, artist)
    if song and song.lyrics:
      return song.lyrics
    return ""
  except Exception as e:
    print(f"❌ Error fetching lyrics for {title} by {artist}: {e}")
    return ""

def detect_language(text):
  try:
    return detect(text)
  except:
    return "unknown"

def translate_to_english(text, source_lang):
  if source_lang == "en":
    return text
  try:
    translator = GoogleTranslator(source=source_lang, target="en")
    return translator.translate(text)
  except Exception as e:
    print(f"❌ Error translating text: {e}")
    return text

def import_playlist(playlist_url: str):
  playlist_id = playlist_url.split("/")[-1].split("?")[0]
  print(f"📥 Importing playlist: {playlist_id}")

  results = sp.playlist_items(playlist_id, limit=100)
  tracks = results["items"]

  while results["next"]:
    results = sp.next(results)
    tracks.extend(results["items"])

  saved = 0
  skipped = 0

  for item in tracks:
    track = item.get("track")
    if not track or not track.get("id"):
      continue

    spotify_id = track["id"]

    if Track.objects.filter(spotify_id=spotify_id).exists():
      print(f"⏭️  Skipping {track['name']} – already in database")
      skipped += 1
      continue
        
    title = track["name"]
    artist = track["artists"][0]["name"]
    album = track["album"]["name"]
    spotify_url = track["external_urls"]["spotify"]

    release_date_str = track["album"].get("release_date", "")
    explicit = track.get("explicit", False)
        
    release_date = None
    if release_date_str:
      try:
        if len(release_date_str) == 4:  # YYYY
          release_date = datetime.strptime(release_date_str, "%Y").date()
        elif len(release_date_str) == 7:  # YYYY-MM
          release_date = datetime.strptime(release_date_str, "%Y-%m").date()
        else:  # YYYY-MM-DD
          release_date = datetime.strptime(release_date_str, "%Y-%m-%d").date()
      except Exception as e:
        print(f"⚠️  Error parsing date '{release_date_str}': {e}")

    # Get lyrics
    lyrics = fetch_lyrics(title, artist)

    if not lyrics.strip():
      print(f"⏭️  Skipping {title} - no lyrics found")
      skipped += 1
      continue

    # Detect language and translate
    lang = detect_language(lyrics)
    lyrics_en = translate_to_english(lyrics, lang)

    # Save to database
    Track.objects.update_or_create(
      spotify_id=track["id"],
      defaults={
        "title": title,
        "artist": artist,
        "album": album,
        "spotify_url": spotify_url,
        "release_date": release_date,
        "explicit": explicit,
        "lyrics": lyrics,
        "lyrics_language": lang,
        "lyrics_en": lyrics_en,
     },
    )
    saved += 1
    print(f"Saved: {title} ({lang})")

  print(f"\n📊 Finished importing playlist. Saved: {saved}, Skipped: {skipped}")

if __name__ == "__main__":
  playlists = [
    #"https://open.spotify.com/playlist/6FllPs6i810p4fhELUDLRw?si=_o6A54UCQ8mhK6FDqvCu8g",
    #"https://open.spotify.com/playlist/1ngtMxQpuF3YD2FQ2l16Ba?si=rBQ76RhuSfC7GDsd89ZWPQ",
    #"https://open.spotify.com/playlist/52ZCVrg8VXkhZkROsDqrAr?si=w81akVd0Qdubx_4wuu884A",
    #"https://open.spotify.com/playlist/0ray47fa8wQc2nDNmpoHYO?si=494LLKLUTLefJ8lA-nxBCg",
    #"https://open.spotify.com/playlist/5DVUEqRL1EV8I9n65eBaAw?si=IlxmBEETQH6v2s388EDJbw",
    #"https://open.spotify.com/playlist/6yYA6aUGp8qUTgQWWYkPkP?si=vLVPK4EnSBamzR_gmoG3MA",
    #"https://open.spotify.com/playlist/63e7OFFUypvmfnLi294JKM?si=5r7ulVSSQUOGFO2ujxFK7A",
    #"https://open.spotify.com/playlist/3CdfoioAm3TCoieKyYPPys?si=Cd-sLJDbR9WfKODFtzzbWA",
    #"https://open.spotify.com/playlist/1i4lngXSiqp1zVWTZgk89C?si=5p-ihpUzQR-7PdCnwQoA_A",
    #"https://open.spotify.com/playlist/414kFQ0Vr3C5ODbaxwq515?si=h1S_aAd7R5q7DUBVrkZXOA",
    #"https://open.spotify.com/playlist/0LoxIwanfEsxi6rb8n8GZb?si=XLLr4S5fR_qkejFcN2KTCw",
    #"https://open.spotify.com/playlist/51C8TAeMej8MZFLJUCINqO?si=KI8bCtcpRaq8quXPYv6DTA",
    #"https://open.spotify.com/playlist/018PAHeJtzX0tmeyuG2HrA?si=pbjkCScRQTeda4nS0BrzlA",
    #"https://open.spotify.com/playlist/5mlMwnp5fhVvxMBf0Xy7AH?si=t_Os5uKYS2GEwrhaYp6N3w",
    #"https://open.spotify.com/playlist/2ZM5ze2pyfmjtnvDOI8EDL?si=u2VlY7ehRMuGRp8Z8DTcUQ"
  ]

  for playlist_url in playlists:
    import_playlist(playlist_url)