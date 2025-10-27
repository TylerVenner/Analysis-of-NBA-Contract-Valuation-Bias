# src/data_processing/cleaning_helpers.py
import unicodedata
import re

def standardize_player_name(name: str) -> str:
    """
    Cleans and standardizes player names so merges work across data sources.
    Examples:
        'LeBron James.' -> 'lebron james'
        'Luka Dončić' -> 'luka doncic'
        'LucMbah a Moute' -> 'luc mbah a moute'
    """
    if not isinstance(name, str):
        return ""

    # Normalize unicode (e.g., accents)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()

    # Replace punctuation and collapse spaces
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name.strip().lower()
