# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Crash-recovery tests for Python-side collection workflows."""

import gc
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from alayalite import Client


class TestRecovery(unittest.TestCase):
    """End-to-end recovery tests that simulate abrupt process exit."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rocksdb_root = os.path.join(self.temp_dir, "RocksDB")
        self._original_rocksdb_dir = os.environ.get("ALAYALITE_ROCKSDB_DIR")
        os.environ["ALAYALITE_ROCKSDB_DIR"] = self.rocksdb_root

    def tearDown(self):
        gc.collect()
        if self._original_rocksdb_dir is None:
            os.environ.pop("ALAYALITE_ROCKSDB_DIR", None)
        else:
            os.environ["ALAYALITE_ROCKSDB_DIR"] = self._original_rocksdb_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _run_crashing_child(self, body: str, exit_code: int = 91) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["ALAYALITE_ROCKSDB_DIR"] = self.rocksdb_root

        script = "\n".join(
            [
                "import os",
                "import numpy as np",
                "from alayalite import Client",
                "",
                f'os.environ["ALAYALITE_ROCKSDB_DIR"] = {self.rocksdb_root!r}',
                f"client = Client({self.temp_dir!r})",
                "",
                textwrap.dedent(body).strip(),
                "",
                f"os._exit({exit_code})",
            ]
        )

        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def test_collection_recovers_after_unclean_exit(self):
        result = self._run_crashing_child(
            """
            coll = client.create_collection("recovering")
            coll.insert(
                [
                    ("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"group": "keep"}),
                    ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"group": "drop"}),
                ]
            )
            coll.upsert(
                [
                    ("a", "Document A v2", np.array([1.0, 0.1, 0.0], dtype=np.float32), {"group": "keep", "v": 2}),
                    ("c", "Document C", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"group": "keep"}),
                ]
            )
            coll.delete_by_id(["b"])
            """,
        )
        self.assertEqual(result.returncode, 91, msg=result.stdout + result.stderr)

        recovery_dir = os.path.join(self.temp_dir, "recovering", "recovery")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "recovering", "schema.json")))
        self.assertTrue(os.path.isfile(os.path.join(recovery_dir, "CURRENT")))
        self.assertTrue(os.path.isdir(os.path.join(recovery_dir, "snapshots")))
        self.assertTrue(os.path.isfile(os.path.join(recovery_dir, "wal.bin")))

        recovered_client = Client(self.temp_dir)
        recovered = recovered_client.get_collection("recovering")

        self.assertIsNotNone(recovered)
        result = recovered.get_by_id(["a", "b", "c"])
        self.assertEqual(result["id"], ["a", "c"])
        self.assertEqual(result["document"][0], "Document A v2")
        self.assertEqual(result["document"][1], "Document C")
        self.assertEqual(result["metadata"][0]["v"], 2)

        del recovered
        del recovered_client
        gc.collect()

    def test_collection_recovery_is_idempotent_across_restarts(self):
        result = self._run_crashing_child(
            """
            coll = client.create_collection("recovering_idempotent")
            coll.insert(
                [
                    ("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"kind": "base"}),
                    ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"kind": "base"}),
                ]
            )
            coll.upsert(
                [("a", "Document A v2", np.array([1.0, 0.2, 0.0], dtype=np.float32), {"kind": "updated"})]
            )
            coll.delete_by_id(["b"])
            """,
        )
        self.assertEqual(result.returncode, 91, msg=result.stdout + result.stderr)

        first_client = Client(self.temp_dir)
        first = first_client.get_collection("recovering_idempotent")
        first_result = first.get_by_id(["a", "b"])
        self.assertEqual(first_result["id"], ["a"])
        self.assertEqual(first_result["document"][0], "Document A v2")
        self.assertEqual(first_result["metadata"][0]["kind"], "updated")

        del first
        del first_client
        gc.collect()

        second_client = Client(self.temp_dir)
        second = second_client.get_collection("recovering_idempotent")
        second_result = second.get_by_id(["a", "b"])
        self.assertEqual(second_result, first_result)

        del second
        del second_client
        gc.collect()

    def test_sq8_collection_recovers_after_unclean_exit(self):
        result = self._run_crashing_child(
            """
            coll = client.create_collection("recovering_sq8", quantization_type="sq8", metric="ip")
            coll.insert(
                [
                    ("x", "Document X", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"kind": "seed"}),
                    ("y", "Document Y", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"kind": "old"}),
                ]
            )
            coll.upsert(
                [("y", "Document Y v2", np.array([0.4, 0.5, 0.7], dtype=np.float32), {"kind": "new"})]
            )
            coll.delete_by_id(["x"])
            """,
        )
        self.assertEqual(result.returncode, 91, msg=result.stdout + result.stderr)

        recovered_client = Client(self.temp_dir)
        recovered = recovered_client.get_collection("recovering_sq8")

        self.assertIsNotNone(recovered)
        result = recovered.get_by_id(["x", "y"])
        self.assertEqual(result["id"], ["y"])
        self.assertEqual(result["document"][0], "Document Y v2")
        self.assertEqual(result["metadata"][0]["kind"], "new")

        del recovered
        del recovered_client
        gc.collect()


if __name__ == "__main__":
    unittest.main()
