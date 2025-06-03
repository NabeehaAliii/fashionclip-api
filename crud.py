# crud.py

from models import SearchHistory
from db import SessionLocal
import json


def save_search_history(user_id, search_type, query_input, results):
    db = SessionLocal()
    entry = SearchHistory(
        user_id=user_id,
        search_type=search_type,
        query_input=query_input,
        top_results=json.dumps(results)  # store as string
    )
    db.add(entry)
    db.commit()
    db.close()

def get_user_history(user_id):
    db = SessionLocal()
    logs = db.query(SearchHistory).filter(
        SearchHistory.user_id == user_id
    ).order_by(SearchHistory.timestamp.desc()).limit(5).all()
    db.close()

    return [json.loads(entry.top_results) for entry in logs]


def clear_user_history(user_id: str):
    db = SessionLocal()
    db.query(SearchHistory).filter(SearchHistory.user_id == user_id).delete()
    db.commit()
    db.close()
