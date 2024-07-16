import json
import sqlite3
from datetime import datetime

from src.const.path import DB_PATH


class InteractionDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    request_method TEXT NOT NULL,
                    request_path TEXT NOT NULL,
                    request_headers TEXT,
                    request_body TEXT,
                    response_status INTEGER,
                    response_headers TEXT,
                    response_body TEXT
                )
            """)
            conn.commit()

    def log_interaction(self, request, response):
        timestamp = datetime.utcnow().isoformat()
        request_method = request.method
        request_path = request.path
        request_headers = json.dumps(dict(request.headers), indent=4)
        try:
            request_body = json.dumps(request.get_json(), indent=4)
        except Exception:
            request_body = request.get_data(as_text=True)
        response_status = response.status_code
        response_headers = json.dumps(dict(response.headers), indent=4)
        try:
            response_body = json.dumps(response.get_json(), indent=4)
        except Exception:
            response_body = response.get_data(as_text=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    timestamp, request_method, request_path, request_headers,
                    request_body, response_status, response_headers, response_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, request_method, request_path, request_headers, request_body,
                  response_status, response_headers, response_body))
            conn.commit()

    def get_interactions(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions")
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
        interactions = []
        for row in rows:
            interaction = dict(zip(columns, row))
            try:
                interaction["request_headers"] = json.loads(interaction["request_headers"])
                interaction["request_body"] = json.loads(interaction["request_body"])
                interaction["response_headers"] = json.loads(interaction["response_headers"])
                interaction["response_body"] = json.loads(interaction["response_body"])
            except json.JSONDecodeError:
                pass
            interactions.append(interaction)
        return interactions
