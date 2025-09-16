import asyncio
import gc
import os
import glob
import statistics
from datetime import datetime
from pathlib import Path
import yaml
import msgspec

from llm_serv.metrics.metrics import ModelMetrics


class LogManager:
    def __init__(self, max_log_length: int = 100, max_log_folder_size_in_mb: int = 1024, models_yaml_path: str = "llm_serv/models.yaml"):
        self._lock = asyncio.Lock()        
        self.max_log_length = max_log_length
        self.max_log_folder_size_in_mb = max_log_folder_size_in_mb
        self.models_yaml_path = models_yaml_path

        self.logs: dict[str, list[ModelMetrics]] = {}  # model_key -> ModelMetrics list
        self._initialized = False

    async def initialize(self):
        """Initialize the LogManager by reading models from YAML and loading latest logs."""
        if not self._initialized:
            await self._initialize_from_disk()
            self._initialized = True

    async def shutdown(self):
        """Shutdown the LogManager by persisting all remaining logs to disk."""
        async with self._lock:
            try:
                print("LogManager shutdown: Archiving remaining logs...")
                # Archive any remaining logs in memory
                await self._archive_memory_logs()
                print(f"LogManager shutdown: Successfully archived logs for {len(self.logs)} models")
            except Exception as e:
                print(f"LogManager shutdown error: {e}")
                # Don't raise exception during shutdown to avoid blocking server shutdown

    async def add_log(self, model_key: str, model_metrics_item: ModelMetrics):
        async with self._lock:
            if model_key not in self.logs:
                self.logs[model_key] = []
            self.logs[model_key].append(model_metrics_item)

            await self.house_keeping()

    async def get_models(self):
        async with self._lock:
            return list(self.logs.keys())

    async def get_logs(
        self, 
        model_key: str, 
        start_time: float | None = None, 
        end_time: float | None = None, 
        limit: int = 100
    ) -> tuple[dict, list[ModelMetrics]]:
        """
        Returns a tuple of (stats, logs) filtered by model_key and start_time and end_time, and limited to limit (latest first, sorted 
        by metric item call_start_time descending)

        If the limit is not reached and/or start_time is not provided / not hit, we will keep reading older logs until we hit a limit
        or run out of logs.

        The stats are computed from the filtered log items on the fly. 

        All CPU heavy ops like reading from disk or stats are done in an asyncio thread.
        """
        async with self._lock:
            # Get in-memory logs for the model
            memory_logs = self.logs.get(model_key, [])
            
            # Filter and sort memory logs
            filtered_logs = []
            for log in memory_logs:
                if start_time is not None and log.call_start_time < start_time:
                    continue
                if end_time is not None and log.call_start_time > end_time:
                    continue
                filtered_logs.append(log)
            
            # Sort by call_start_time descending (latest first)
            filtered_logs.sort(key=lambda x: x.call_start_time, reverse=True)
            
            # If we have enough logs or no archived logs, return what we have
            if len(filtered_logs) >= limit:
                result_logs = filtered_logs[:limit]
                stats = await self._run_in_thread(self.get_stats, result_logs)
                return stats, result_logs
            
            # Need to read from disk to get more logs
            archived_logs = await self._read_archived_logs(model_key, start_time, end_time, limit - len(filtered_logs))
            
            # Combine and sort all logs
            all_logs = filtered_logs + archived_logs
            all_logs.sort(key=lambda x: x.call_start_time, reverse=True)
            
            # Limit the results
            result_logs = all_logs[:limit]
            
            # Compute stats in a separate thread
            stats = await self._run_in_thread(self.get_stats, result_logs)
            
            return stats, result_logs

    def get_stats(self, data_points: list[ModelMetrics]) -> dict:
        """
        Computes statistics for a given model key and data points.

        Will return a dictionary with the following keys:
        - average_duration
        - median_duration
        - max_duration
        - min_duration
        - std_duration

        - average_tokens_per_second
        - median_tokens_per_second
        - max_tokens_per_second
        - min_tokens_per_second
        - std_tokens_per_second
        
        - percent_success
        - status_counter (status_code, count)

        - average_internal_retries
        
        """
        if not data_points:
            return {
                "average_duration": 0,
                "median_duration": 0,
                "max_duration": 0,
                "min_duration": 0,
                "std_duration": 0,
                "average_tokens_per_second": 0,
                "median_tokens_per_second": 0,
                "max_tokens_per_second": 0,
                "min_tokens_per_second": 0,
                "std_tokens_per_second": 0,
                "percent_success": 0,
                "status_counter": {},
                "average_internal_retries": 0,
                "total_requests": 0
            }
        
        # Extract values for statistics
        durations = [dp.call_duration for dp in data_points if dp.call_duration > 0]
        tokens_per_second = [dp.tokens_per_second for dp in data_points if dp.tokens_per_second > 0]
        internal_retries = [dp.internal_retries for dp in data_points]
        
        # Success tracking
        successful_requests = sum(1 for dp in data_points if dp.status_code is not None and 200 <= dp.status_code < 300)
        total_requests = len(data_points)
        
        # Status code counter
        status_counter = {}
        for dp in data_points:
            if dp.status_code is not None:
                status_counter[dp.status_code] = status_counter.get(dp.status_code, 0) + 1
        
        # Duration statistics
        duration_stats = {
            "average_duration": statistics.mean(durations) if durations else 0,
            "median_duration": statistics.median(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
        }
        
        # Tokens per second statistics
        tps_stats = {
            "average_tokens_per_second": statistics.mean(tokens_per_second) if tokens_per_second else 0,
            "median_tokens_per_second": statistics.median(tokens_per_second) if tokens_per_second else 0,
            "max_tokens_per_second": max(tokens_per_second) if tokens_per_second else 0,
            "min_tokens_per_second": min(tokens_per_second) if tokens_per_second else 0,
            "std_tokens_per_second": statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0,
        }
        
        stats = {
            **duration_stats,
            **tps_stats,
            "percent_success": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "status_counter": status_counter,
            "average_internal_retries": statistics.mean(internal_retries) if internal_retries else 0,
            "total_requests": total_requests
        }
        
        return stats

    async def house_keeping(self):
        """
        Housekeeping is done to keep the logs from growing too large.
        If the log length is greater than the max_log_length, we need to pack, save to disk and unload memory.
        Also checks if total log folder size exceeds max_log_folder_size_in_mb and cleans up if needed.

        Log name convention is: metrics/model_key/YYYYMMDDHHMMSS-YYYYMMDDHHMMSS.json (start_time-end_time).
        """
        # Calculate total log length
        log_length = sum(len(v) for v in self.logs.values())
        
        # Archive logs if memory limit is exceeded
        if log_length > self.max_log_length:
            await self._archive_memory_logs()
        
        # Check total folder size and cleanup if needed
        await self._cleanup_by_folder_size()
        
        # Force garbage collection
        gc.collect()
    
    async def _archive_memory_logs(self):
        """Archive logs from memory to disk."""
        for model_key, model_logs in list(self.logs.items()):
            if not model_logs:
                continue
                
            # Sanitize model_key for filesystem
            safe_model_key = self._sanitize_filename(model_key)
            metrics_dir = f"metrics/{safe_model_key}"
            
            # Ensure output folder exists
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Sort logs by start time
            sorted_logs = sorted(model_logs, key=lambda x: x.call_start_time)
            
            if sorted_logs:
                # Create filename with start and end times
                start_time = sorted_logs[0].call_start_time
                end_time = sorted_logs[-1].call_start_time
                
                start_str = datetime.fromtimestamp(start_time).strftime("%Y%m%d%H%M%S")
                end_str = datetime.fromtimestamp(end_time).strftime("%Y%m%d%H%M%S")
                
                filename = f"{metrics_dir}/{start_str}-{end_str}.json"
                
                # Serialize logs to JSON using msgspec
                await self._run_in_thread(self._write_logs_to_file, filename, sorted_logs)
                
                # Clear memory logs for this model
                self.logs[model_key] = []
    
    async def _cleanup_by_folder_size(self):
        """Clean up old log files if total folder size exceeds limit."""
        try:
            # Calculate total size of all log files
            total_size_mb = await self._calculate_total_log_folder_size()
            
            if total_size_mb <= self.max_log_folder_size_in_mb:
                return
            
            # Get all log files across all models, sorted by modification time (oldest first)
            all_log_files = await self._get_all_log_files_sorted_by_age()
            
            # Delete files until we're under the size limit
            for file_path in all_log_files:
                if total_size_mb <= self.max_log_folder_size_in_mb:
                    break
                
                try:
                    file_size_mb = await self._run_in_thread(self._get_file_size_mb, file_path)
                    await self._run_in_thread(os.remove, file_path)
                    total_size_mb -= file_size_mb
                    print(f"Deleted old log file: {file_path} ({file_size_mb:.2f} MB)")
                except Exception as e:
                    print(f"Error deleting log file {file_path}: {e}")
                    
        except Exception as e:
            print(f"Error during folder size cleanup: {e}")
    
    async def _calculate_total_log_folder_size(self) -> float:
        """Calculate total size of all log files in MB."""
        def calculate_size():
            total_size = 0
            metrics_base_dir = "metrics"
            
            if not os.path.exists(metrics_base_dir):
                return 0
            
            for root, _dirs, files in os.walk(metrics_base_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            continue
            
            return total_size / (1024 * 1024)  # Convert to MB
        
        return await self._run_in_thread(calculate_size)
    
    async def _get_all_log_files_sorted_by_age(self) -> list[str]:
        """Get all log files across all models, sorted by modification time (oldest first)."""
        def get_files():
            all_files = []
            metrics_base_dir = "metrics"
            
            if not os.path.exists(metrics_base_dir):
                return []
            
            for root, _dirs, files in os.walk(metrics_base_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
            
            # Sort by modification time (oldest first)
            all_files.sort(key=os.path.getmtime)
            return all_files
        
        return await self._run_in_thread(get_files)
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            return 0

    async def _initialize_from_disk(self):
        """Initialize logs from disk by reading models.yaml and loading latest log files."""
        try:
            # Read models from models.yaml
            model_keys = await self._get_model_keys_from_yaml()
            
            async with self._lock:
                # Initialize empty logs for all models
                for model_key in model_keys:
                    if model_key not in self.logs:
                        self.logs[model_key] = []
                
                # Load latest logs from disk for each model
                for model_key in model_keys:
                    await self._load_latest_logs_for_model(model_key)
                    
        except Exception as e:
            print(f"Error during initialization from disk: {e}")
    
    async def _get_model_keys_from_yaml(self) -> list[str]:
        """Read model keys from models.yaml file."""
        try:
            yaml_path = Path(self.models_yaml_path)
            if not yaml_path.exists():
                print(f"Models YAML file not found at {self.models_yaml_path}")
                return []
            
            def read_yaml():
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    models_section = data.get('MODELS', {})
                    return list(models_section.keys())
            
            return await self._run_in_thread(read_yaml)
            
        except Exception as e:
            print(f"Error reading models.yaml: {e}")
            return []
    
    async def _load_latest_logs_for_model(self, model_key: str):
        """Load the latest log files for a specific model into memory."""
        try:
            safe_model_key = self._sanitize_filename(model_key)
            metrics_dir = f"metrics/{safe_model_key}"
            
            if not os.path.exists(metrics_dir):
                return
            
            # Get the latest log file for this model
            pattern = f"{metrics_dir}/*.json"
            archived_files = glob.glob(pattern)
            
            if not archived_files:
                return
            
            # Sort by modification time and get the latest file
            archived_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = archived_files[0]
            
            # Load logs from the latest file
            logs = await self._run_in_thread(self._read_logs_from_file, latest_file)
            
            # Add to memory, keeping only the most recent logs up to max_log_length
            if logs:
                # Sort by call_start_time and keep the latest ones
                logs.sort(key=lambda x: x.call_start_time, reverse=True)
                self.logs[model_key].extend(logs[:self.max_log_length])
                
        except Exception as e:
            print(f"Error loading logs for model {model_key}: {e}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by replacing unsafe characters with underscores."""
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        return filename

    async def _run_in_thread(self, func, *args):
        """Run a CPU-intensive function in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    def _write_logs_to_file(self, filename: str, logs: list[ModelMetrics]):
        """Write logs to file using msgspec JSON encoding."""
        with open(filename, 'wb') as f:
            # Encode logs as JSON bytes
            data = msgspec.json.encode(logs)
            f.write(data)

    async def _read_archived_logs(
        self, 
        model_key: str, 
        start_time: float | None, 
        end_time: float | None, 
        limit: int
    ) -> list[ModelMetrics]:
        """Read archived logs from disk with filtering."""
        safe_model_key = self._sanitize_filename(model_key)
        metrics_dir = f"metrics/{safe_model_key}"
        
        if not os.path.exists(metrics_dir):
            return []
        
        # Get all archived log files sorted by modification time (newest first)
        pattern = f"{metrics_dir}/*.json"
        archived_files = glob.glob(pattern)
        archived_files.sort(key=os.path.getmtime, reverse=True)
        
        collected_logs = []
        
        for file_path in archived_files:
            if len(collected_logs) >= limit:
                break
                
            try:
                file_logs = await self._run_in_thread(self._read_logs_from_file, file_path)
                
                # Filter logs by time
                for log in file_logs:
                    if len(collected_logs) >= limit:
                        break
                    if start_time is not None and log.call_start_time < start_time:
                        continue
                    if end_time is not None and log.call_start_time > end_time:
                        continue
                    collected_logs.append(log)
                        
            except Exception as e:
                # Log error and continue with other files
                print(f"Error reading archived log file {file_path}: {e}")
                continue
        
        # Sort by call_start_time descending
        collected_logs.sort(key=lambda x: x.call_start_time, reverse=True)
        return collected_logs[:limit]

    def _read_logs_from_file(self, filename: str) -> list[ModelMetrics]:
        """Read logs from file using msgspec JSON decoding."""
        try:
            with open(filename, 'rb') as f:
                data = f.read()
                return msgspec.json.decode(data, type=list[ModelMetrics])
        except Exception as e:
            print(f"Error decoding log file {filename}: {e}")
            return []
