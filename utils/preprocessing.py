import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# pobieranie zasobow NLTK
for res in ['punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(
            f'tokenizers/{res}' if 'punkt' in res
            else f'sentiment/{res}' if 'vader' in res
            else f'corpora/{res}' if res in ['stopwords', 'wordnet']
            else f'taggers/{res}'
        )
    except LookupError:
        nltk.download(res)

# -------- NOISE / EMOTION WORDS ----------------------------

# zbior slow charakterystycznych dla piosenek ktore nic nie mowia o nastroju, wszystkie beda usuniete podczas preprocessingu
LYRICS_NOISE_WORDS = {
    'yeah', 'yea', 'oh', 'ooh', 'ah', 'uh', 'hey', 'la', 'na', 'da',
    'oo', 'woah', 'whoa', 'hmm', 'shh', 'ohe', 'ay', 'yo',
    'mm', 'woo', 'eh', 'ayy', 'ai', 'ta', 'ma', 'em',
    'ya', 'yah', 'oye', 'aye', 'nah', 'mhm', 'huh',
    'baby', 'babe', 'come', 'got', 'get', 'gonna', 'gon',
    'wanna', 'wan', 'gotta', 'know', 'like', 'take', 'make',
    'say', 'said', 'tell', 'told', 'give', 'gave', 'put',
    'back', 'next', 'another', 'enough', 'much',
    'chorus', 'verse', 'bridge', 'outro', 'intro',
    'find', 'share', 'song', 'hop', 'lo', 'ho',
}

# zbior slow ktore bezposrednio wskazuja na nastroj i musza byc zachowane nawet jesli normalnie byly uznane za stopwords
EMOTION_WORDS = {
    'love', 'happy', 'joy', 'smile', 'laugh', 'beautiful', 'wonderful',
    'amazing', 'bright', 'shine', 'light', 'hope', 'dream', 'free',
    'celebrate', 'dance', 'party', 'fun', 'enjoy', 'paradise',
    'sad', 'cry', 'pain', 'hurt', 'dark', 'death', 'die', 'lost',
    'broken', 'tear', 'tears', 'alone', 'lonely', 'fear', 'hate',
    'wrong', 'fail', 'failure', 'suffer', 'suffering', 'grief',
    'fire', 'burn', 'wild', 'crazy', 'run', 'fast', 'loud', 'scream',
    'rock', 'fight', 'power', 'fury', 'rage', 'thunder', 'storm',
    'calm', 'peace', 'quiet', 'gentle', 'soft', 'slow',
    'sleep', 'night', 'rest', 'breathe', 'flow', 'whisper', 'soul',
    'not', 'never', 'no', 'nothing', 'nobody', 'nowhere', 'none',
    'sorry', 'wish', 'anymore', 'lie', 'forever', 'heart',
    'sunshine', 'moon', 'stranger', 'higher',
}

# -------- PREPROCESSOR ----------------------------

class LyricsPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()   # tworzy lematyzator
        base_stops = set(stopwords.words('english'))    # pobiera standardowa liste ok. 180 angielskich stopwords z NLTK
        self.stop_words = (base_stops | LYRICS_NOISE_WORDS) - EMOTION_WORDS # buduje ostateczna liste slow do usuniecia

    # czysci surowy tekst przed tokenizacja
    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text) # usuwa linki internetowe
        text = re.sub(r"[^\w\s']", ' ', text)   # usuwa znaki specjalne oprocz liter, cyfr, spacji, apostrofu
        text = re.sub(r'\d+', '', text)         # usuwa cyfry
        text = re.sub(r'\s+', ' ', text).strip()    # zmienia wielokrotne spacje na jedna i usuwa space z poczatku i konca
        return text

    # przetwarza jedna piosenke
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)

        processed = []

        for token in tokens:
            if len(token) < 2:
                continue

            # jesli slowo jest emocjonalne - dodaje do wyniku i przechodzi do nastepnego
            if token in EMOTION_WORDS:
                processed.append(token)
                continue

            # jesli slowo jest na liscie stopwords - pomija je
            if token in self.stop_words:
                continue

            token = self.lemmatizer.lemmatize(token, pos='v')   # lematyzuje jako czasownik
            token = self.lemmatizer.lemmatize(token, pos='n')   # lematyzuje jako rzeczownik
            processed.append(token)

        return ' '.join(processed)

    # przetwarza caly dataset
    def preprocess_dataframe(self, df, text_column='LYRICS_EN'):
        print(f"Preprocessing {len(df)} texts...")
        df = df.copy()
        df['PROCESSED_LYRICS'] = df[text_column].apply(self.preprocess)
        df['WORD_COUNT'] = df['PROCESSED_LYRICS'].apply(lambda x: len(x.split()))
        print(f"Average word count: {df['WORD_COUNT'].mean():.1f} | Median: {df['WORD_COUNT'].median():.1f}")
        return df

# -------- VADER — ekstraktor cech + relabelowanie ----------------------------

class VADERFeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    # wyciaga 7 cech VADER z jednej piosenki
    def extract(self, text):
        if pd.isna(text) or text.strip() == "":
            return np.zeros(7)

        sentences = [s.strip() for s in text.split('\n') if s.strip()]  # dzieli tekst na linijki
        if not sentences:
            return np.zeros(7)

        scores = [self.analyzer.polarity_scores(s) for s in sentences]  # dla kazdej linijki VADER oblicza slownik z czterema wartosciami (pos, neg, neu, compound)
        compounds = [s['compound'] for s in scores]    # wyciaga same wartosci compound (ogolny sentyment) ze wszystkich linijek
        total = len(scores)
        return np.array([
            np.mean([s['pos'] for s in scores]),    # 1. sredni poziom pozytywnosci
            np.mean([s['neg'] for s in scores]),    # 2. sredni poziom negatywnosci
            np.mean([s['neu'] for s in scores]),    # 3. sredni poziom neutralnosci
            np.mean(compounds),                     # 4. sredni ogolny sentyment calej piosenki
            sum(1 for s in scores if s['compound'] >= 0.05) / total,    # 5. procent pozytywnych linijek
            sum(1 for s in scores if s['compound'] <= -0.05) / total,   # 6. procent negatywnych linijek
            np.std(compounds),                      # 7. jak bardzo sentyment skacze miedzy linijkami
        ])

    # wyciaga cechy VADER dla calego datasetu
    def extract_dataframe(self, df, text_column='LYRICS_EN'):
        print("Extracting VADER features...")
        vader_features = np.array(df[text_column].apply(self.extract).tolist())
        cols = ['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound',
                'vader_pos_ratio', 'vader_neg_ratio', 'vader_compound_std']
        return pd.DataFrame(vader_features, columns=cols, index=df.index)
