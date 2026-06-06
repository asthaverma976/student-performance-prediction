"""
MongoDB Handler for Student Performance Prediction System.
All operations are wrapped in try/except so the app works without MongoDB.
"""

import datetime


class MongoHandler:
    """
    Handles MongoDB operations for storing student data and predictions.
    Gracefully degrades if MongoDB (pymongo) is not installed or the server is unreachable.
    """

    def __init__(self, uri: str = "mongodb://localhost:27017/",
                 db_name: str = "student_performance_db"):
        """
        Initialize the MongoDB connection.

        Args:
            uri: MongoDB connection URI.
            db_name: Name of the database to use.
        """
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        self.connect()

    def connect(self):
        """Establish connection to MongoDB. Fails silently if unavailable."""
        try:
            from pymongo import MongoClient
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            # Force a connection attempt to verify the server is reachable
            self.client.server_info()
            self.db = self.client[self.db_name]
            self.connected = True
            print("✅ Connected to MongoDB successfully.")
        except ImportError:
            print("⚠️  pymongo not installed. Running without MongoDB.")
            self.connected = False
        except Exception as e:
            print(f"⚠️  Could not connect to MongoDB: {e}")
            print("   Running without database support.")
            self.connected = False

    def insert_students(self, student_records: list) -> int:
        """
        Insert student records into the 'students' collection.

        Args:
            student_records: List of dicts, each representing a student.

        Returns:
            Number of records inserted, or 0 if DB is unavailable.
        """
        if not self.connected:
            print("⚠️  MongoDB not available. Skipping student data insertion.")
            return 0

        try:
            collection = self.db["students"]
            # Clear existing data to avoid duplicates on re-run
            collection.delete_many({})
            result = collection.insert_many(student_records)
            count = len(result.inserted_ids)
            print(f"✅ Inserted {count} student records into MongoDB.")
            return count
        except Exception as e:
            print(f"❌ Error inserting students: {e}")
            return 0

    def get_students(self, limit: int = 0) -> list:
        """
        Retrieve student records from the 'students' collection.

        Args:
            limit: Maximum number of records to retrieve. 0 = all records.

        Returns:
            List of student record dicts.
        """
        if not self.connected:
            return []

        try:
            collection = self.db["students"]
            cursor = collection.find({}, {"_id": 0})
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"❌ Error retrieving students: {e}")
            return []

    def insert_prediction(self, input_data: dict, predicted_score: float,
                          performance_category: str) -> bool:
        """
        Store a prediction result in the 'predictions' collection.

        Args:
            input_data: Dict of input features used for the prediction.
            predicted_score: The predicted performance score.
            performance_category: Category label (Excellent/Good/Average/Poor).

        Returns:
            True if insertion succeeded, False otherwise.
        """
        if not self.connected:
            return False

        try:
            collection = self.db["predictions"]
            record = {
                "input_features": input_data,
                "predicted_score": round(predicted_score, 2),
                "performance_category": performance_category,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
            collection.insert_one(record)
            return True
        except Exception as e:
            print(f"❌ Error inserting prediction: {e}")
            return False

    def get_predictions(self, limit: int = 50) -> list:
        """
        Retrieve recent predictions from the 'predictions' collection.

        Args:
            limit: Maximum number of predictions to return.

        Returns:
            List of prediction dicts, most recent first.
        """
        if not self.connected:
            return []

        try:
            collection = self.db["predictions"]
            cursor = (collection.find({}, {"_id": 0})
                      .sort("timestamp", -1)
                      .limit(limit))
            return list(cursor)
        except Exception as e:
            print(f"❌ Error retrieving predictions: {e}")
            return []

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                print("🔌 MongoDB connection closed.")
            except Exception:
                pass


if __name__ == "__main__":
    # Quick connectivity test
    handler = MongoHandler()
    if handler.connected:
        print(f"Database: {handler.db_name}")
        print(f"Collections: {handler.db.list_collection_names()}")
    else:
        print("MongoDB is not available — the app will still work without it.")
    handler.close()
