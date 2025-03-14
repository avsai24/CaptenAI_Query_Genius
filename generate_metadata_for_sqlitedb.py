import os
import sqlite3
import json

DATABASE_FOLDER = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/sqlite_databases"

def get_sqlite_databases(directory):
    """Retrieve all SQLite database files from a given directory"""
    return [f for f in os.listdir(directory) if f.endswith(".db")]

def get_database_metadata(db_path):
    """Extract metadata (tables & columns) from a SQLite database"""
    metadata = {"database": os.path.basename(db_path), "tables": {}}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            metadata["tables"][table_name] = [
                {"column_name": col[1], "data_type": col[2]} for col in columns
            ]

        conn.close()
    except Exception as e:
        metadata["error"] = str(e)

    return metadata

def get_all_sqlite_metadata(directory):
    databases = get_sqlite_databases(directory)
    all_metadata = []

    for db in databases:
        db_path = os.path.join(directory, db)
        metadata = get_database_metadata(db_path)
        all_metadata.append(metadata)

    return all_metadata

metadata_info = get_all_sqlite_metadata(DATABASE_FOLDER)

print(json.dumps(metadata_info, indent=4))