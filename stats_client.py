#!/usr/bin/env python
# Stats client for hybrid distributed training
# Connects to the stats server and sends real-time metrics

import asyncio
import websockets
import json
import logging
import time
import threading
import os
from typing import Dict, List, Any, Optional, Union
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stats_client")

class StatsClient:
    """Client for sending stats to the stats server"""
    
    def __init__(self, server_url="ws://localhost:8765", node_id="coordinator", reconnect_delay=5):
        """
        Initialize the stats client.
        
        Args:
            server_url: WebSocket URL of the stats server
            node_id: ID of this node
            reconnect_delay: Delay in seconds before reconnecting after disconnection
        """
        self.server_url = server_url
        self.node_id = node_id
        self.reconnect_delay = reconnect_delay
        self.websocket = None
        self.connected = False
        self.running = False
        self.send_queue = Queue()
        self.client_thread = None
        self.lock = threading.Lock()
        
        # Stats buffer for when disconnected
        self.stats_buffer = []
        self.max_buffer_size = 1000
        
        logger.info(f"Stats client initialized with server_url={server_url}, node_id={node_id}")
    
    def start(self):
        """Start the stats client in a background thread"""
        if self.running:
            logger.warning("Stats client already running")
            return
        
        self.running = True
        self.client_thread = threading.Thread(target=self._run_client_loop, daemon=True)
        self.client_thread.start()
        logger.info("Stats client started")
    
    def stop(self):
        """Stop the stats client"""
        self.running = False
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join(timeout=5)
        logger.info("Stats client stopped")
    
    def _run_client_loop(self):
        """Run the client loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                loop.run_until_complete(self._connect_and_process())
            except Exception as e:
                logger.error(f"Error in client loop: {e}", exc_info=True)
            
            if self.running:
                time.sleep(self.reconnect_delay)
        
        loop.close()
    
    async def _connect_and_process(self):
        """Connect to the server and process messages"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                self.connected = True
                logger.info(f"Connected to stats server at {self.server_url}")
                
                # Send buffered stats
                await self._send_buffered_stats()
                
                # Send initial registration
                await self._send_registration()
                
                # Process any queued messages
                while self.running:
                    # Check if there are messages to send
                    if not self.send_queue.empty():
                        message = self.send_queue.get()
                        await websocket.send(json.dumps(message))
                        self.send_queue.task_done()
                    
                    # Wait for a short time before checking again
                    await asyncio.sleep(0.01)
        
        except (websockets.exceptions.ConnectionClosed, 
                websockets.exceptions.WebSocketException, 
                ConnectionRefusedError) as e:
            logger.warning(f"Connection to stats server lost: {e}")
        finally:
            self.connected = False
            self.websocket = None
    
    async def _send_registration(self):
        """Send registration message to the server"""
        registration = {
            "type": "worker_heartbeat",
            "worker_id": self.node_id,
            "status": "active",
            "timestamp": time.time(),
            "info": {
                "pid": os.getpid(),
                "hostname": os.uname().nodename,
                "node_type": "coordinator" if self.node_id == "coordinator" else "worker"
            }
        }
        
        await self.websocket.send(json.dumps(registration))
    
    async def _send_buffered_stats(self):
        """Send buffered stats to the server"""
        if not self.stats_buffer:
            return
        
        logger.info(f"Sending {len(self.stats_buffer)} buffered stats messages")
        
        with self.lock:
            for stats in self.stats_buffer:
                await self.websocket.send(json.dumps(stats))
            
            self.stats_buffer = []
    
    def send_worker_stats(self, worker_id: str, stats: Dict[str, Any]):
        """
        Send worker stats to the server.
        
        Args:
            worker_id: ID of the worker
            stats: Stats to send
        """
        message = {
            "type": "worker_stats",
            "worker_id": worker_id,
            "stats": stats,
            "timestamp": time.time()
        }
        
        self._queue_or_buffer_message(message)
    
    def send_aggregated_stats(self, stats: Dict[str, Any]):
        """
        Send aggregated stats to the server.
        
        Args:
            stats: Aggregated stats to send
        """
        message = {
            "type": "aggregated_stats",
            "stats": stats,
            "timestamp": time.time()
        }
        
        self._queue_or_buffer_message(message)
    
    def send_heartbeat(self, status: str = "active"):
        """
        Send heartbeat to the server.
        
        Args:
            status: Status of this node
        """
        message = {
            "type": "worker_heartbeat",
            "worker_id": self.node_id,
            "status": status,
            "timestamp": time.time()
        }
        
        self._queue_or_buffer_message(message)
    
    def _queue_or_buffer_message(self, message: Dict[str, Any]):
        """Queue a message or buffer it if disconnected"""
        if self.connected:
            self.send_queue.put(message)
        else:
            with self.lock:
                self.stats_buffer.append(message)
                
                # Trim buffer if too large
                if len(self.stats_buffer) > self.max_buffer_size:
                    self.stats_buffer = self.stats_buffer[-self.max_buffer_size:]

class WorkerMetricsCollector:
    """
    Collects metrics from workers and aggregates them.
    Integrates with StatsClient to send metrics to the stats server.
    """
    def __init__(self, stats_client: Optional[StatsClient] = None, heartbeat_interval: int = 10):
        """
        Initialize the worker metrics collector.
        
        Args:
            stats_client: StatsClient instance (optional)
            heartbeat_interval: Interval in seconds for sending heartbeats
        """
        self.stats_client = stats_client
        self.heartbeat_interval = heartbeat_interval
        self.worker_metrics = {}
        self.aggregated_metrics = {
            "loss": 0.0,
            "tokens": 0,
            "steps": 0,
            "workers": 0,
            "start_time": time.time()
        }
        self.last_heartbeat = 0
        self.last_metrics_update = 0
        self.lock = threading.Lock()
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            metrics: Metrics to update
        """
        with self.lock:
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = {
                    "first_seen": time.time(),
                    "last_update": time.time(),
                    "update_count": 0
                }
            
            worker_data = self.worker_metrics[worker_id]
            worker_data.update(metrics)
            worker_data["last_update"] = time.time()
            worker_data["update_count"] += 1
        
        # Send to stats server if available
        if self.stats_client:
            self.stats_client.send_worker_stats(worker_id, self.worker_metrics[worker_id])
        
        # Update aggregated metrics if needed
        self._update_aggregated_metrics()
        
        # Send heartbeat if needed
        current_time = time.time()
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            self._send_heartbeat()
            self.last_heartbeat = current_time
    
    def update_aggregated_metrics(self, metrics: Dict[str, Any]):
        """
        Update aggregated metrics directly.
        
        Args:
            metrics: Metrics to update
        """
        with self.lock:
            self.aggregated_metrics.update(metrics)
            self.aggregated_metrics["last_update"] = time.time()
        
        # Send to stats server if available
        if self.stats_client:
            self.stats_client.send_aggregated_stats(self.aggregated_metrics)
        
        # Send heartbeat if needed
        current_time = time.time()
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            self._send_heartbeat()
            self.last_heartbeat = current_time
    
    def _update_aggregated_metrics(self):
        """Update aggregated metrics based on worker metrics"""
        current_time = time.time()
        
        # Only update periodically to avoid excessive computation
        if current_time - self.last_metrics_update < 1.0:
            return
        
        with self.lock:
            # Count active workers (updated in the last 30 seconds)
            active_workers = 0
            total_loss = 0.0
            total_tokens = 0
            max_steps = 0
            
            for worker_id, metrics in self.worker_metrics.items():
                if current_time - metrics.get("last_update", 0) < 30:
                    active_workers += 1
                    
                    # Accumulate metrics if available
                    if "loss" in metrics:
                        total_loss += metrics["loss"]
                    if "tokens" in metrics:
                        total_tokens += metrics["tokens"]
                    if "steps" in metrics:
                        max_steps = max(max_steps, metrics["steps"])
            
            # Update aggregated metrics
            self.aggregated_metrics["workers"] = active_workers
            
            if active_workers > 0:
                self.aggregated_metrics["loss"] = total_loss / active_workers
            
            self.aggregated_metrics["tokens"] = total_tokens
            self.aggregated_metrics["steps"] = max_steps
            self.aggregated_metrics["last_update"] = current_time
            
            # Calculate tokens per second
            runtime = current_time - self.aggregated_metrics["start_time"]
            if runtime > 0:
                self.aggregated_metrics["tokens_per_second"] = total_tokens / runtime
        
        # Send to stats server if available
        if self.stats_client:
            self.stats_client.send_aggregated_stats(self.aggregated_metrics)
        
        self.last_metrics_update = current_time
    
    def _send_heartbeat(self):
        """Send heartbeat to stats server"""
        if self.stats_client:
            self.stats_client.send_heartbeat()

# Example usage
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Test the stats client")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="Stats server URL")
    parser.add_argument("--node-id", type=str, default=f"test_node_{random.randint(1000, 9999)}", help="Node ID")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between updates (seconds)")
    args = parser.parse_args()
    
    # Create stats client
    client = StatsClient(server_url=args.server, node_id=args.node_id)
    client.start()
    
    # Create metrics collector
    collector = WorkerMetricsCollector(stats_client=client)
    
    try:
        # Send metrics periodically
        for i in range(1000):
            # Update worker metrics
            worker_metrics = {
                "loss": random.random() * 10,
                "tokens": random.randint(100, 1000),
                "steps": i,
                "memory_used": random.randint(1000, 10000),
                "gpu_util": random.randint(0, 100)
            }
            collector.update_worker_metrics(args.node_id, worker_metrics)
            
            # Sleep
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        client.stop()