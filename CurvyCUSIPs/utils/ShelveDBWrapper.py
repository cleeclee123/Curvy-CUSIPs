import shelve
import os 

class ShelveDBWrapper:
    def __init__(self, db_path, create=False):
        self.db_path = db_path
        self.db = None
        if not create:
            if not os.path.exists(self.db_path):
                raise ValueError("DB does not exist")

    def open(self):
        """Open the shelve database."""
        self.db = shelve.open(self.db_path)

    def close(self):
        """Close the shelve database."""
        if self.db is not None:
            self.db.close()

    def get(self, key):
        """Retrieve a value by key."""
        if self.db is not None:
            return self.db[key]
        raise RuntimeError("Database is not open.")

    def set(self, key, value):
        """Set a value by key."""
        if self.db is not None:
            self.db[key] = value
        else:
            raise RuntimeError("Database is not open.")

    def keys(self):
        if self.db is not None:
            return self.db.keys() 
        else:
            raise RuntimeError("Database is not open.")
        
    def exists(self, key):
        """Check if a key exists in the database."""
        if self.db is not None:
            return key in self.db
        else:
            raise RuntimeError("Database is not open.")
        
    def delete(self, key):
        """Delete a key from the database."""
        if self.db is not None:
            if key in self.db:
                del self.db[key]
            else:
                raise KeyError(f"Key '{key}' does not exist in the database.")
        else:
            raise RuntimeError("Database is not open.")