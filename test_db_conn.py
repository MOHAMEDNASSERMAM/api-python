  import psycopg2
import os

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "jobito",
    "user": "postgres",
    "password": "mlpoknbv"
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("SUCCESS: Connection to PostgreSQL successful!")
    conn.close()
except Exception as e:
    print(f"FAILURE: {e}")
