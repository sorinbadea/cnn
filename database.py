#!/usr/bin/env python3
import psycopg2
import sys
import filters

class DataBaseInterface():
    def __init__(self, hostname, database, username, password, port):
        self._host = hostname
        self._database = database
        self._user = username
        self._password = password
        self._port = port
        self._connection = None
        self._cursor = None
        self._trained_data = {}

    def __enter__(self):
        if self.database_connect() is False:
            print(f"❌ Error connecting to database")
            sys.exit(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
         self.database_disconnect()

    def database_connect(self):
        try:
            self._connection = psycopg2.connect(
                host=self._host,
                database=self._database,
                user=self._user,
                password=self._password,
                port=self._port,
                connect_timeout=5
            )
            self._cursor = self._connection.cursor()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def database_disconnect(self):
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()

    def get_data(self, table_name):
        try:
            self._cursor.execute("SELECT samples FROM " + table_name)
            rows = self._cursor.fetchall()
            return rows
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def insert_data(self, table_name, data):
        try:
            self._cursor.execute("INSERT INTO " + table_name + " (samples) VALUES (%s)", (data,))
            self._connection.commit()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        return True

    def create_table(self, table_name):
        try:
            drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
            self._cursor.execute(drop_table_query)
            print(f"✅ Table '{table_name}' dropped successfully.")
            create_table_query = f"CREATE TABLE {table_name} (samples DOUBLE PRECISION[]);"
            print(f"✅ Table '{table_name}' created successfully.")
            self._cursor.execute(create_table_query)
            self._connection .commit()
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        return True

    def load_trained_data(self):
        """
        Loads the all trained data from filters.py into memory
        """
        for shape_index in range(len(filters.shapes)):
            shape = filters.shapes[shape_index]
            for key in shape['filters']:
                self._trained_data[key] = self.get_data(key)
        print("✅ Trained data loaded successfully.")

    def get_trained_data(self, key):
        """
        Returns the trained data for a specific kernel key
        @param key: kernel key
        """
        return self._trained_data[key] or None

if __name__ == "__main__":
    db = DataBaseInterface('localhost','myapp','postgres','password',5432)
    db.database_connect()
    res = db.get_data('digit_1_filter_1')
    for row in res:
        print("res=", row)
    db.database_disconnect()