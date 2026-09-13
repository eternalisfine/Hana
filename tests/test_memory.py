import unittest
from unittest.mock import patch, MagicMock
import memory
from datetime import datetime
import json
import sqlite3

class TestMemory(unittest.TestCase):
    def setUp(self):
        # Create an in-memory database for testing
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        # Override the _conn context manager to yield our in-memory connection
        self.conn_patcher = patch('memory._conn')
        self.mock_conn_cm = self.conn_patcher.start()
        
        # We need a context manager that yields self.conn
        class MockCM:
            def __init__(self, conn):
                self.conn = conn
            def __enter__(self):
                return self.conn
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        self.mock_conn_cm.return_value = MockCM(self.conn)
        
        # Initialize schema
        memory.init_db()

    def tearDown(self):
        self.conn.close()
        self.conn_patcher.stop()

    def test_add_and_get_messages(self):
        memory.add_message("session_1", "user", "こんにちは")
        memory.add_message("session_1", "assistant", "こんにちは！元気ですか？")
        
        messages = memory.get_recent_messages(10)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "こんにちは")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "こんにちは！元気ですか？")

    def test_vocabulary_tracking(self):
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=False)
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=True)
        memory.upsert_vocabulary("猫", "ねこ", "NOUN", produced=True)
        
        vocab = memory.get_recent_vocabulary(10)
        self.assertEqual(len(vocab), 1)
        self.assertEqual(vocab[0]["word"], "猫")
        self.assertEqual(vocab[0]["times_seen"], 3)
        self.assertEqual(vocab[0]["times_produced"], 2)

    def test_streak_calculation(self):
        # By default, streak should be 0 if no messages
        self.assertEqual(memory.get_streak(), 0)
        # Adding activity with messages should increase streak
        memory.log_daily_activity(messages=1)
        self.assertEqual(memory.get_streak(), 1)

if __name__ == "__main__":
    unittest.main()
