import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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


# ==============================================================================
# NOISE / EMOTION WORDS
# ==============================================================================
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


# ==============================================================================
# PREPROCESSOR
# ==============================================================================
class LyricsPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        base_stops = set(stopwords.words('english'))
        self.stop_words = (base_stops | LYRICS_NOISE_WORDS) - EMOTION_WORDS

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        processed = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token in EMOTION_WORDS:
                processed.append(token)
                continue
            if token in self.stop_words:
                continue
            token = self.lemmatizer.lemmatize(token, pos='v')
            token = self.lemmatizer.lemmatize(token, pos='n')
            processed.append(token)
        return ' '.join(processed)

    def preprocess_dataframe(self, df, text_column='LYRICS_EN'):
        print(f"Preprocessing {len(df)} texts...")
        df = df.copy()
        df['PROCESSED_LYRICS'] = df[text_column].apply(self.preprocess)
        df['WORD_COUNT'] = df['PROCESSED_LYRICS'].apply(lambda x: len(x.split()))
        print(f"Average word count: {df['WORD_COUNT'].mean():.1f} | Median: {df['WORD_COUNT'].median():.1f}")
        return df


# ==============================================================================
# VADER — ekstraktor cech + relabelowanie
# ==============================================================================
class VADERFeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract(self, text):
        if pd.isna(text) or text.strip() == "":
            return np.zeros(7)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        if not sentences:
            return np.zeros(7)
        scores = [self.analyzer.polarity_scores(s) for s in sentences]
        compounds = [s['compound'] for s in scores]
        total = len(scores)
        return np.array([
            np.mean([s['pos'] for s in scores]),
            np.mean([s['neg'] for s in scores]),
            np.mean([s['neu'] for s in scores]),
            np.mean(compounds),
            sum(1 for s in scores if s['compound'] >= 0.05) / total,
            sum(1 for s in scores if s['compound'] <= -0.05) / total,
            np.std(compounds),
        ])

    def extract_dataframe(self, df, text_column='LYRICS_EN'):
        print("Extracting VADER features...")
        vader_features = np.array(df[text_column].apply(self.extract).tolist())
        cols = ['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound',
                'vader_pos_ratio', 'vader_neg_ratio', 'vader_compound_std']
        return pd.DataFrame(vader_features, columns=cols, index=df.index)



