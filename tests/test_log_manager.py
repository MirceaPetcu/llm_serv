import asyncio
import os
import shutil
import tempfile
import unittest
import time
from unittest import IsolatedAsyncioTestCase

from llm_serv.metrics.log_manager import LogManager
from llm_serv.metrics.metrics import ModelMetrics


class TestLogManager(IsolatedAsyncioTestCase):
    """Comprehensive test suite for LogManager with cleanup."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize LogManager with test-friendly settings
        self.log_manager = LogManager(max_log_length=5, max_log_archive_files=3)
        
        # Sample metrics for testing
        self.sample_metrics = [
            ModelMetrics(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                call_start_time=time.time() - 100,
                call_end_time=time.time() - 95,
                call_duration=5.0,
                tokens_per_second=30.0,
                status_code=200,
                internal_retries=0
            ),
            ModelMetrics(
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                call_start_time=time.time() - 90,
                call_end_time=time.time() - 85,
                call_duration=5.0,
                tokens_per_second=60.0,
                status_code=200,
                internal_retries=1
            ),
            ModelMetrics(
                input_tokens=150,
                output_tokens=75,
                total_tokens=225,
                call_start_time=time.time() - 80,
                call_end_time=time.time() - 75,
                call_duration=5.0,
                tokens_per_second=45.0,
                status_code=500,
                internal_retries=2
            )
        ]

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_add_log_and_get_models(self):
        """Test adding logs and retrieving model keys."""
        # Initially no models
        models = await self.log_manager.get_models()
        self.assertEqual(models, [])
        
        # Add logs for different models
        await self.log_manager.add_log("model_a", self.sample_metrics[0])
        await self.log_manager.add_log("model_b", self.sample_metrics[1])
        await self.log_manager.add_log("model_a", self.sample_metrics[2])
        
        # Check models are tracked
        models = await self.log_manager.get_models()
        self.assertIn("model_a", models)
        self.assertIn("model_b", models)
        self.assertEqual(len(models), 2)

    async def test_get_stats_empty_data(self):
        """Test get_stats with empty data."""
        stats = self.log_manager.get_stats([])
        
        expected_keys = {
            "average_duration", "median_duration", "max_duration", "min_duration", "std_duration",
            "average_tokens_per_second", "median_tokens_per_second", "max_tokens_per_second", 
            "min_tokens_per_second", "std_tokens_per_second",
            "percent_success", "status_counter", "average_internal_retries", "total_requests"
        }
        
        self.assertEqual(set(stats.keys()), expected_keys)
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["percent_success"], 0)

    async def test_get_stats_with_data(self):
        """Test get_stats with actual data."""
        stats = self.log_manager.get_stats(self.sample_metrics)
        
        # Check basic stats
        self.assertEqual(stats["total_requests"], 3)
        self.assertAlmostEqual(stats["percent_success"], 66.67, places=1)  # 2 out of 3 successful
        self.assertEqual(stats["average_duration"], 5.0)
        self.assertEqual(stats["median_duration"], 5.0)
        
        # Check status counter
        self.assertEqual(stats["status_counter"][200], 2)
        self.assertEqual(stats["status_counter"][500], 1)
        
        # Check tokens per second stats
        self.assertEqual(stats["average_tokens_per_second"], 45.0)  # (30 + 60 + 45) / 3
        self.assertEqual(stats["max_tokens_per_second"], 60.0)
        self.assertEqual(stats["min_tokens_per_second"], 30.0)
        
        # Check internal retries
        self.assertEqual(stats["average_internal_retries"], 1.0)  # (0 + 1 + 2) / 3

    async def test_get_logs_memory_only(self):
        """Test get_logs when all logs are in memory."""
        # Add logs
        for i, metric in enumerate(self.sample_metrics):
            await self.log_manager.add_log("test_model", metric)
        
        # Get all logs
        stats, logs = await self.log_manager.get_logs("test_model")
        
        self.assertEqual(len(logs), 3)
        self.assertEqual(stats["total_requests"], 3)
        
        # Logs should be sorted by call_start_time descending (latest first)
        self.assertGreaterEqual(logs[0].call_start_time, logs[1].call_start_time)
        self.assertGreaterEqual(logs[1].call_start_time, logs[2].call_start_time)

    async def test_get_logs_with_time_filtering(self):
        """Test get_logs with start_time and end_time filtering."""
        # Add logs
        for metric in self.sample_metrics:
            await self.log_manager.add_log("test_model", metric)
        
        # Filter by start time (should get only the latest 2)
        # Use the second metric's start time as the filter
        start_time = self.sample_metrics[1].call_start_time
        stats, logs = await self.log_manager.get_logs("test_model", start_time=start_time)
        
        # Should get 2 logs (the second and third metrics)
        self.assertEqual(len(logs), 2)
        for log in logs:
            self.assertGreaterEqual(log.call_start_time, start_time)

    async def test_get_logs_with_limit(self):
        """Test get_logs with limit parameter."""
        # Add logs
        for metric in self.sample_metrics:
            await self.log_manager.add_log("test_model", metric)
        
        # Get only 2 logs
        stats, logs = await self.log_manager.get_logs("test_model", limit=2)
        
        self.assertEqual(len(logs), 2)
        self.assertEqual(stats["total_requests"], 2)

    async def test_housekeeping_archiving(self):
        """Test housekeeping and log archiving."""
        # Add enough logs to trigger archiving (max_log_length = 5)
        for i in range(6):
            metric = ModelMetrics(
                input_tokens=100 + i,
                output_tokens=50 + i,
                total_tokens=150 + i,
                call_start_time=time.time() - (100 - i * 10),
                call_end_time=time.time() - (95 - i * 10),
                call_duration=5.0 + i,
                tokens_per_second=30.0 + i,
                status_code=200,
                internal_retries=i
            )
            await self.log_manager.add_log("test_model", metric)
        
        # Check that housekeeping was triggered and logs were archived
        self.assertTrue(os.path.exists("metrics"))
        self.assertTrue(os.path.exists("metrics/test_model"))
        
        # Memory should be cleared
        models = await self.log_manager.get_models()
        if "test_model" in models:
            # If model still exists, its log list should be empty or much smaller
            self.assertLessEqual(len(self.log_manager.logs.get("test_model", [])), 1)

    async def test_filename_sanitization(self):
        """Test filename sanitization for unsafe model keys."""
        unsafe_model_key = "model/with:unsafe*chars?"
        metric = self.sample_metrics[0]
        
        await self.log_manager.add_log(unsafe_model_key, metric)
        
        # Trigger archiving
        for i in range(5):
            await self.log_manager.add_log(unsafe_model_key, metric)
        
        # Check that sanitized directory was created
        safe_key = self.log_manager._sanitize_filename(unsafe_model_key)
        self.assertTrue(os.path.exists(f"metrics/{safe_key}"))
        self.assertEqual(safe_key, "model_with_unsafe_chars_")

    async def test_archive_cleanup(self):
        """Test cleanup of old archive files."""
        model_key = "test_model"
        
        # Create more archive files than max_log_archive_files (3)
        for i in range(5):
            # Add logs to trigger archiving
            for j in range(6):  # Exceed max_log_length
                metric = ModelMetrics(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    call_start_time=time.time() - (1000 + i * 100 + j * 10),
                    call_end_time=time.time() - (995 + i * 100 + j * 10),
                    call_duration=5.0,
                    tokens_per_second=30.0,
                    status_code=200,
                    internal_retries=0
                )
                await self.log_manager.add_log(model_key, metric)
        
        # Wait a bit for file operations
        await asyncio.sleep(0.1)
        
        # Check that only max_log_archive_files (3) remain
        if os.path.exists(f"metrics/{model_key}"):
            archive_files = os.listdir(f"metrics/{model_key}")
            json_files = [f for f in archive_files if f.endswith('.json')]
            self.assertLessEqual(len(json_files), 3)

    async def test_archived_log_reading(self):
        """Test reading logs from archived files."""
        model_key = "test_model"
        
        # Add logs and trigger archiving
        original_metrics = []
        for i in range(6):
            metric = ModelMetrics(
                input_tokens=100 + i,
                output_tokens=50 + i,
                total_tokens=150 + i,
                call_start_time=time.time() - (100 - i * 10),
                call_end_time=time.time() - (95 - i * 10),
                call_duration=5.0 + i,
                tokens_per_second=30.0 + i,
                status_code=200,
                internal_retries=i
            )
            original_metrics.append(metric)
            await self.log_manager.add_log(model_key, metric)
        
        # Wait for archiving to complete
        await asyncio.sleep(0.1)
        
        # Try to get logs - should read from archive
        stats, logs = await self.log_manager.get_logs(model_key, limit=10)
        
        # Should be able to retrieve some logs from archive
        self.assertGreater(len(logs), 0)
        self.assertGreater(stats["total_requests"], 0)

    async def test_concurrent_operations(self):
        """Test concurrent log operations."""
        async def add_logs_for_model(model_key: str, count: int):
            for i in range(count):
                metric = ModelMetrics(
                    input_tokens=100 + i,
                    output_tokens=50 + i,
                    total_tokens=150 + i,
                    call_start_time=time.time() - i,
                    call_end_time=time.time() - i + 1,
                    call_duration=1.0,
                    tokens_per_second=150.0,
                    status_code=200,
                    internal_retries=0
                )
                await self.log_manager.add_log(model_key, metric)
        
        # Run concurrent operations
        tasks = [
            add_logs_for_model("model_1", 3),
            add_logs_for_model("model_2", 3),
            add_logs_for_model("model_3", 3)
        ]
        
        await asyncio.gather(*tasks)
        
        # Check that all models were added
        models = await self.log_manager.get_models()
        self.assertGreaterEqual(len(models), 3)

    async def test_error_handling_corrupted_files(self):
        """Test error handling when reading corrupted archive files."""
        model_key = "test_model"
        
        # Create metrics directory and add a corrupted file
        os.makedirs(f"metrics/{model_key}", exist_ok=True)
        with open(f"metrics/{model_key}/corrupted.json", "w") as f:
            f.write("invalid json content")
        
        # Should handle corrupted files gracefully
        try:
            stats, logs = await self.log_manager.get_logs(model_key)
            # Should return empty results without crashing
            self.assertEqual(len(logs), 0)
        except Exception as e:
            self.fail(f"get_logs should handle corrupted files gracefully, but raised: {e}")


if __name__ == "__main__":
    unittest.main()
