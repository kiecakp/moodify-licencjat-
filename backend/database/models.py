from django.db import models

class Track(models.Model):
  # Metadata from Spotify API
  spotify_id = models.CharField(max_length=64, unique=True)
  title = models.CharField(max_length=200)
  artist = models.CharField(max_length=200)
  album = models.CharField(max_length=200, blank=True, null=True)
  spotify_url = models.URLField(blank=True, null=True)
  release_date = models.DateField(
    max_length=20,
    blank=True,
    null=True,
    help_text="YYYY, YYYY-MM, or YYYY-MM-DD from Spotify API"
  )
    
  explicit = models.BooleanField(default=False)

  # Song lyrics
  lyrics = models.TextField(blank=True)
  lyrics_language = models.CharField(
    max_length=10,
    blank=True,
    help_text="Language code of the lyrics, e.g., 'en' for English"
  )
  lyrics_en = models.TextField(
    blank=True,
    help_text="English translation of the lyrics, if available"
  )

  # Mood tag (completed by ML model)
  mood_tag = models.CharField(
    max_length=50,
    blank=True,
    help_text="Predicted by ML model (happy, sad, chill, energetic...)"
  )

  # Metadata timestamps
  created_at = models.DateTimeField(auto_now_add=True)

  def __str__(self):
    return f"{self.title} by {self.artist}"