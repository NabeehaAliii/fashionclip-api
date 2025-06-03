# models.py

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    search_type = Column(String)
    query_input = Column(Text)
    top_results = Column(Text)  # JSON string of results
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
