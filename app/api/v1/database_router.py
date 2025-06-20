from fastapi import APIRouter, HTTPException, Depends
import logging
from ...schemas.match import Match
from typing import List
import sqlite3
from contextlib import contextmanager
from .authenticator import admin_only_auth
import logging

logger = logging.getLogger(__name__)

@contextmanager
def open_db(db_path: str):
    """ Custom context manager to handle the closing of database if something goes wrong. """
    connection = sqlite3.connect(db_path)

    try:
        cursor = connection.cursor()
        yield cursor
    except sqlite3.DatabaseError as error:
        logger.error(f'The following error occurred: {error}')
    finally:
        connection.commit()
        connection.close()

class DatabaseRouter:
    def __init__(self):
        _path: str = 'database.db'

    def get_router(self) -> APIRouter:
        router = APIRouter(prefix="/database", tags=["Database"], dependencies=[Depends(admin_only_auth)])

        @router.get("/get-all-matches", response_model=List[Match])
        def get_all_matches() -> dict:
            return self.get_all_matches()

        return router
    

    def _create_table(self) -> None:
        with open_db(self._path) as cursor:
            try:
                cursor.execute("""CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    division TEXT,
                    league TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    playing_field TEXT,
                    scheduled_match_time TEXT""")
                
                logging.info("Table created successfuly")
            except Exception as e:
                logging.error(f"Error creating table: {e}")

    def insert_match(self, division: str, league: str, home_team: str, away_team: str, playing_field: str, scheduled_match_time) -> bool:
        """ Inserts data in the database """

        # Make sure the table exist before trying to insert data.
        self._create_table()

        try:
            with open_db(self._path) as cursor:
                cursor.execute("""INSERT INTO matches (division, league, home_team, away_team, playing_field, scheduled_match_time) VALUES (?, ?, ?, ?, ?, ?)""",
                               (division, league, home_team, away_team, playing_field, scheduled_match_time))

                return True

        except Exception as e:
            # Log error or handle appropriately
            logging.error(f"Error inserting match: {e}")

            return False
        
    def get_all_matches(self):
        """ Fetches all matches from the database """

        try:
            with open_db(self._path) as cursor:
                cursor.execute("SELECT * FROM matches")
                return cursor.fetchall()  # Returns a list of tuples

        except Exception as e:
            logging.error(f"Error retrieving matches: {e}")
            return []
