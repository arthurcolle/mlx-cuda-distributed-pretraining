#!/usr/bin/env python
# Stats server for hybrid distributed training
# Provides real-time stats collection via WebSockets

import asyncio
import websockets
import json
import logging
import argparse
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stats_server.log")
    ]
)
logger = logging.getLogger("stats_server")

class StatsServer:
    """WebSocket server for real-time stats collection"""
    
    def __init__(self, host="0.0.0.0", port=8765, save_dir="./logs/stats"):
        """
        Initialize the stats server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            save_dir: Directory to save stats to
        """
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Connected clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Store stats by worker ID
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Last aggregated stats
        self.aggregated_stats: Dict[str, Any] = {}
        
        # Stats history (limited to last 1000 updates)
        self.stats_history: List[Dict[str, Any]] = []
        self.max_history_length = 1000
        
        # Run ID for this session
        self.run_id = f"{int(time.time())}"
        
        # Server state
        self.is_running = False
        self.start_time = time.time()
        
        logger.info(f"Stats server initialized with host={host}, port={port}")
    
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send initial data
        await self._send_initial_data(websocket)
    
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def _send_initial_data(self, websocket: websockets.WebSocketServerProtocol):
        """Send initial data to a new client"""
        # Send server info
        server_info = {
            "type": "server_info",
            "run_id": self.run_id,
            "start_time": self.start_time,
            "uptime": time.time() - self.start_time,
            "worker_count": len(self.worker_stats),
            "client_count": len(self.clients)
        }
        await websocket.send(json.dumps(server_info))
        
        # Send latest worker stats
        for worker_id, stats in self.worker_stats.items():
            worker_data = {
                "type": "worker_stats",
                "worker_id": worker_id,
                "stats": stats
            }
            await websocket.send(json.dumps(worker_data))
        
        # Send latest aggregated stats
        if self.aggregated_stats:
            await websocket.send(json.dumps({
                "type": "aggregated_stats",
                "stats": self.aggregated_stats
            }))
        
        # Send recent history (last 50 entries max)
        history_subset = self.stats_history[-50:] if len(self.stats_history) > 50 else self.stats_history
        await websocket.send(json.dumps({
            "type": "stats_history",
            "history": history_subset
        }))
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            return
        
        message_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_json) for client in self.clients],
            return_exceptions=True
        )
    
    async def handle_stats(self, websocket: websockets.WebSocketServerProtocol):
        """Handle stats updates from workers"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    if message_type == "worker_stats":
                        await self._handle_worker_stats(data)
                    elif message_type == "aggregated_stats":
                        await self._handle_aggregated_stats(data)
                    elif message_type == "worker_heartbeat":
                        await self._handle_worker_heartbeat(data)
                    elif message_type == "get_stats":
                        await self._handle_get_stats(websocket, data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        finally:
            await self.unregister(websocket)
    
    async def _handle_worker_stats(self, data: Dict[str, Any]):
        """Handle worker stats update"""
        worker_id = data.get("worker_id")
        stats = data.get("stats", {})
        timestamp = data.get("timestamp", time.time())
        
        if not worker_id:
            logger.warning("Missing worker_id in worker_stats message")
            return
        
        # Update worker stats
        self.worker_stats[worker_id] = {
            **stats,
            "last_update": timestamp
        }
        
        # Add to history
        history_entry = {
            "timestamp": timestamp,
            "worker_id": worker_id,
            "stats": stats
        }
        self._add_to_history(history_entry)
        
        # Broadcast update
        await self.broadcast({
            "type": "worker_stats_update",
            "worker_id": worker_id,
            "stats": self.worker_stats[worker_id]
        })
        
        # Save to disk periodically
        if timestamp % 60 < 1:  # Approximately once per minute
            self._save_stats()
    
    async def _handle_aggregated_stats(self, data: Dict[str, Any]):
        """Handle aggregated stats update"""
        stats = data.get("stats", {})
        timestamp = data.get("timestamp", time.time())
        
        # Update aggregated stats
        self.aggregated_stats = {
            **stats,
            "last_update": timestamp
        }
        
        # Add to history
        history_entry = {
            "timestamp": timestamp,
            "type": "aggregated",
            "stats": stats
        }
        self._add_to_history(history_entry)
        
        # Broadcast update
        await self.broadcast({
            "type": "aggregated_stats_update",
            "stats": self.aggregated_stats
        })
        
        # Save to disk periodically
        if timestamp % 60 < 1:  # Approximately once per minute
            self._save_stats()
    
    async def _handle_worker_heartbeat(self, data: Dict[str, Any]):
        """Handle worker heartbeat"""
        worker_id = data.get("worker_id")
        status = data.get("status", "unknown")
        timestamp = data.get("timestamp", time.time())
        
        if not worker_id:
            logger.warning("Missing worker_id in worker_heartbeat message")
            return
        
        # Update worker status
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id]["status"] = status
            self.worker_stats[worker_id]["last_heartbeat"] = timestamp
        else:
            self.worker_stats[worker_id] = {
                "status": status,
                "last_heartbeat": timestamp
            }
        
        # Broadcast heartbeat
        await self.broadcast({
            "type": "worker_heartbeat",
            "worker_id": worker_id,
            "status": status,
            "timestamp": timestamp
        })
    
    async def _handle_get_stats(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle request for stats"""
        stats_type = data.get("stats_type", "all")
        
        if stats_type == "all":
            await self._send_initial_data(websocket)
        elif stats_type == "worker":
            worker_id = data.get("worker_id")
            if worker_id and worker_id in self.worker_stats:
                await websocket.send(json.dumps({
                    "type": "worker_stats",
                    "worker_id": worker_id,
                    "stats": self.worker_stats[worker_id]
                }))
        elif stats_type == "aggregated":
            await websocket.send(json.dumps({
                "type": "aggregated_stats",
                "stats": self.aggregated_stats
            }))
        elif stats_type == "history":
            limit = data.get("limit", 50)
            history_subset = self.stats_history[-limit:] if len(self.stats_history) > limit else self.stats_history
            await websocket.send(json.dumps({
                "type": "stats_history",
                "history": history_subset
            }))
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """Add an entry to the stats history"""
        self.stats_history.append(entry)
        
        # Trim history if needed
        if len(self.stats_history) > self.max_history_length:
            self.stats_history = self.stats_history[-self.max_history_length:]
    
    def _save_stats(self):
        """Save stats to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save worker stats
        worker_stats_file = self.save_dir / f"worker_stats_{self.run_id}_{timestamp}.json"
        with open(worker_stats_file, 'w') as f:
            json.dump(self.worker_stats, f, indent=2)
        
        # Save aggregated stats
        agg_stats_file = self.save_dir / f"aggregated_stats_{self.run_id}_{timestamp}.json"
        with open(agg_stats_file, 'w') as f:
            json.dump(self.aggregated_stats, f, indent=2)
        
        # Save recent history (last 100 entries)
        history_subset = self.stats_history[-100:] if len(self.stats_history) > 100 else self.stats_history
        history_file = self.save_dir / f"stats_history_{self.run_id}_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(history_subset, f, indent=2)
    
    async def serve(self):
        """Start the stats server"""
        self.is_running = True
        self.start_time = time.time()
        
        # Create the stats directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        async with websockets.serve(
            self.handle_stats, self.host, self.port
        ):
            logger.info(f"Stats server running at ws://{self.host}:{self.port}")
            
            # Keep server running
            while self.is_running:
                await asyncio.sleep(1)
                
                # Check for stale workers (no heartbeat in 30 seconds)
                current_time = time.time()
                stale_workers = []
                
                for worker_id, stats in self.worker_stats.items():
                    last_heartbeat = stats.get("last_heartbeat", 0)
                    if current_time - last_heartbeat > 30:
                        stats["status"] = "stale"
                        stale_workers.append(worker_id)
                
                if stale_workers:
                    logger.warning(f"Stale workers detected: {stale_workers}")
                    for worker_id in stale_workers:
                        await self.broadcast({
                            "type": "worker_status_change",
                            "worker_id": worker_id,
                            "status": "stale",
                            "timestamp": current_time
                        })

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Stats server for hybrid distributed training")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--save-dir", type=str, default="./logs/stats", help="Directory to save stats to")
    args = parser.parse_args()
    
    # Create server
    server = StatsServer(
        host=args.host,
        port=args.port,
        save_dir=args.save_dir
    )
    
    # Start server
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Stats server stopped by user")
    except Exception as e:
        logger.error(f"Error in stats server: {e}", exc_info=True)

if __name__ == "__main__":
    main()