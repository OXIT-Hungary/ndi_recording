import os
from fastapi import APIRouter, HTTPException, Depends
import logging
from ...schemas.match import Match
from typing import List
import sqlite3
from sqlite3 import OperationalError
from contextlib import contextmanager
from .authenticator import admin_only_auth
import logging
from ...schemas.stream_start_request import StreamStartRequest
from ...schemas.match import Match

logger = logging.getLogger(__name__)

@contextmanager
def open_db(db_path: str):
    """ Custom context manager to handle the closing of database if something goes wrong. """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    try:
        yield cursor  # exceptions inside the with-block are handled by the caller
    finally:
        connection.commit()
        connection.close()

class DatabaseRouter:
    def __init__(self):
        pass

    def get_router(self) -> APIRouter:
        router = APIRouter(prefix="/database", tags=["Database"], dependencies=[Depends(admin_only_auth)])

        @router.get("/get-all-matches", response_model=List[Match])
        def get_all_matches() -> dict:
            return Database.get_all_matches()

        return router
    

class Database:

    _database_path = 'metadata.db'
    _txt_path = 'metadata.txt'

    @staticmethod
    def _create_table() -> None:
        with open_db(Database._database_path) as cursor:
            try:
                cursor.execute("""CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    division TEXT,
                    league TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    playing_field TEXT,
                    scheduled_match_time TEXT,
                    UNIQUE(division, league, home_team, away_team, playing_field, scheduled_match_time));""")
                
                logging.info("Table created successfuly")
            except Exception as e:
                logging.error(f"Error creating table: {e}")

    #def insert_match(division: str, league: str, home_team: str, away_team: str, playing_field: str, scheduled_match_time) -> bool:
    def insert_match(payload: StreamStartRequest) -> bool:
        """ Inserts data in the database """

        # Make sure the table exist before trying to insert data.
        Database._create_table()

        try:
            with open_db(Database._database_path) as cursor:
                cursor.execute("""INSERT OR IGNORE INTO matches 
                    (division, league, home_team, away_team, playing_field, scheduled_match_time) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (payload.division, payload.league, payload.home_team, payload.away_team, payload.playing_field, payload.scheduled_match_time))

                return True

        except Exception as e:
            # Log error or handle appropriately
            logging.error(f"Error inserting match: {e}")

            return False

    @staticmethod        
    def get_all_matches() -> list[Match]:
        """ Fetches all matches from the database """
        matches = []

        try:
            with open_db(Database._database_path) as cursor:
                if cursor is None:
                    logging.error("Database connection failed.")
                    return []
        
                cursor.execute("SELECT * FROM matches")

                data = cursor.fetchall()

                matches = [Match(
                    id=row[0],
                    division=row[1],
                    league=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    playing_field=row[5],
                    scheduled_match_time=row[6]
                ) for row in data]

                return matches  # Returns a list of tuples

        except OperationalError as e:
            if "no such table" in str(e):
                logging.warning("Table 'matches' does not exist.")
                return []
            else:
                logging.error(f"Database operational error: {e}")
                return []

        except Exception as e:
            logging.error(f"Unexpected error retrieving matches: {e}")
            return []
        
    @staticmethod
    def save_to_txt(payload: StreamStartRequest) -> bool:
        try:
            # get the keys for the fields
            fields = list(payload.model_fields.keys())
            # Get values for those fields, except the last value.
            values = [getattr(payload, key, '') for key in fields[:-1]]

            file_exists = os.path.exists(Database._txt_path)

            with open(Database._txt_path, 'a') as f:
                if not file_exists:
                    # Write header row, except the last value.
                    f.write('\t'.join([key.upper() for key in fields[:-1]]) + '\n')

                # Write values row.
                f.write('\t'.join(str(v) if v is not None else '' for v in values) + '\n')

            return True  # Successful write
        
        except Exception as e:
            print(f"Error saving row: {e}")

            return False
