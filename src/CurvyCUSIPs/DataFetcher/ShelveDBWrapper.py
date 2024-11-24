import shelve


class ShelveDBWrapper:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = None

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
