import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as dateparse
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def fetch_instagram_events(club=None, start_date=None, end_date=None):
    sample_posts = [
        {
            "caption": "Join the AI Club meetup on July 5th at 6 PM! Location: Engineering Building, Room 101.",
            "date_posted": "2025-06-20",
            "instagram_post_url": "https://instagram.com/p/ABC123",
        },
        {
            "caption": "Chess Club tournament happening July 10th, 4 PM. Register now!",
            "date_posted": "2025-06-22",
            "instagram_post_url": "https://instagram.com/p/XYZ789",
        },
    ]

    filtered_events = []

    for post in sample_posts:
        # If club filter is set, simple check if club name is in caption (case insensitive)
        if club and club.lower() not in post["caption"].lower():
            continue

        # Extract event date from caption (naive approach for now)
        event_date = extract_event_date(post["caption"])
        if not event_date:
            continue

        # Filter by date range if provided
        if start_date and event_date < start_date:
            continue
        if end_date and event_date > end_date:
            continue

        event = {
            "club": club if club else "Unknown Club",
            "event_date": event_date.isoformat(),
            "details": post["caption"],
            "post_url": post["instagram_post_url"]
        }
        filtered_events.append(event)

    return filtered_events


def extract_event_date(caption):
    """
    Use NLP and dateutil to extract the first date mentioned in caption.
    Return a datetime.date object or None.
    """
    sentences = sent_tokenize(caption)
    for sent in sentences:
        try:
            # Attempt to parse any date from the sentence
            dt = dateparse(sent, fuzzy=True)
            return dt.date()
        except Exception:
            continue
    return None
