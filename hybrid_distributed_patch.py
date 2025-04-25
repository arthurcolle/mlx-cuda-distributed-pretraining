#!/usr/bin/env python
# Patch for integrating stats collection into hybrid_distributed.py

# Add these imports to the top of hybrid_distributed.py
"""
from stats_client import StatsClient, WorkerMetricsCollector
"""

# Add this after the logger initialization in the main function (around line 951)
"""
    # Initialize stats client if requested
    stats_client = None
    metrics_collector = None
    if args.stats_server:
        logger.info(f"Connecting to stats server at {args.stats_server}")
        stats_client = StatsClient(
            server_url=args.stats_server,
            node_id=f"coordinator_{run_id}"
        )
        stats_client.start()
        
        metrics_collector = WorkerMetricsCollector(stats_client=stats_client)
"""

# Add this to the argument parser in main() (around line 924)
"""
    parser.add_argument("--stats-server", type=str, default=None, 
                       help="WebSocket URL of the stats server (e.g. ws://localhost:8765)")
    parser.add_argument("--stats-interval", type=int, default=10,
                       help="Interval in seconds for aggregating and sending stats")
"""

# Modify the training loop in main() (around line 1022) to collect stats
"""
                # Forward and backward pass
                loss, tokens, grads = device_manager.parallel_forward_backward(
                    model, batch[:, :-1], batch[:, 1:]
                )
                
                # Update model
                optimizer.update(model, grads)
                
                # Send metrics to stats collector if available
                if metrics_collector:
                    # Update aggregated metrics
                    metrics_collector.update_aggregated_metrics({
                        "loss": float(loss),
                        "tokens": int(tokens),
                        "steps": step,
                        "learning_rate": float(optimizer.learning_rate.item())
                    })
                    
                    # Also report per-device metrics
                    for device_id, device_stats in device_manager.get_device_stats().items():
                        metrics_collector.update_worker_metrics(
                            device_id, 
                            {
                                "loss": float(device_stats.get("loss", 0.0)),
                                "tokens": int(device_stats.get("tokens", 0)),
                                "steps": step,
                                "device_type": device_stats.get("device_type", "unknown")
                            }
                        )
"""

# Add this to the finally block in main() (around line 1055) to clean up
"""
        # Clean up stats client
        if 'stats_client' in locals() and stats_client:
            stats_client.stop()
"""

# Add a method to HybridDeviceManager class to collect device stats (around line 560)
"""
    def get_device_stats(self):
        """Get stats from all devices"""
        device_stats = {}
        
        # Local MLX devices
        for device_name in self.mlx_devices:
            device_stats[device_name] = {
                "device_type": "mlx",
                "status": "active"
            }
        
        # Remote workers
        for worker_id, connector in self.remote_connections.items():
            try:
                status = connector.check_status()
                device_stats[worker_id] = {
                    "device_type": "remote",
                    "status": status.get("status", "unknown"),
                    "gpu_info": status.get("gpu_info", {})
                }
            except Exception as e:
                logger.warning(f"Error getting status for worker {worker_id}: {e}")
                device_stats[worker_id] = {
                    "device_type": "remote",
                    "status": "error",
                    "error": str(e)
                }
        
        return device_stats
"""

# Modify the _compute_forward_backward method in HybridDeviceManager to track per-device stats (around line 530)
"""
        # Compute backward pass
        grad_fn = mx.grad(lambda m, x, y: nn.losses.cross_entropy(m(x), y).sum())
        gradients = grad_fn(model, inputs, targets)
        
        # Track per-device statistics if available
        device_name = str(mx.default_device())
        device_stats = {
            "loss": loss.sum() / ntoks,
            "tokens": ntoks,
            "device_type": device_name
        }
        
        # Store in device specific data (accessed via get_device_stats)
        if hasattr(self, "_device_specific_stats"):
            self._device_specific_stats[device_name] = device_stats
        else:
            self._device_specific_stats = {device_name: device_stats}
        
        return loss.sum() / ntoks, ntoks, gradients
"""

# Modify the run_hybrid_distributed.sh script to add stats server support
"""
# Add after line 31 in run_hybrid_distributed.sh
echo "Starting stats server..."
python stats_server.py --host 0.0.0.0 --port 8765 --save-dir "$LOG_DIR/stats" > "$LOG_DIR/stats_server.log" 2>&1 &
STATS_PID=$!
echo "Stats server started with PID: $STATS_PID"
sleep 2  # Give server time to start

# Add --stats-server flag to hybrid_distributed.py call (line 78)
python hybrid_distributed.py \\
  --config "$CONFIG" \\
  --workers "$WORKERS" \\
  --data-dir "$DATA_DIR" \\
  --run-id "$RUN_ID" \\
  --stats-server "ws://localhost:8765" > "$LOG_DIR/hybrid_training.log" 2>&1 &

# Add to final echo message (line 108)
echo "Stats server: http://localhost:8765/dashboard.html"
"""

# Add a simple web dashboard for the stats
"""
# Create a file at dashboard.html in the same directory as stats_server.py
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-box {
            background-color: #e9f5ff;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }
        .worker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .worker-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 15px;
        }
        .worker-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .worker-name {
            font-weight: bold;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-active {
            background-color: #4CAF50;
        }
        .status-stale {
            background-color: #FFC107;
        }
        .status-error {
            background-color: #F44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hybrid Training Dashboard</h1>
            <div>
                <span id="connection-status">Connecting...</span>
                <button id="reconnect-btn" style="display: none;">Reconnect</button>
            </div>
        </div>
        
        <div class="card">
            <h2>Training Progress</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Loss</div>
                    <div class="stat-value" id="current-loss">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Step</div>
                    <div class="stat-value" id="current-step">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Tokens Processed</div>
                    <div class="stat-value" id="tokens-processed">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Tokens/second</div>
                    <div class="stat-value" id="tokens-per-second">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Active Workers</div>
                    <div class="stat-value" id="active-workers">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Runtime</div>
                    <div class="stat-value" id="runtime">-</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Loss History</h2>
            <div class="chart-container">
                <canvas id="loss-chart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Throughput History</h2>
            <div class="chart-container">
                <canvas id="throughput-chart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>Worker Status</h2>
            <div class="worker-grid" id="worker-container">
                <!-- Worker cards will be added here -->
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let socket;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        const reconnectInterval = 5000; // 5 seconds
        
        // Charts
        let lossChart;
        let throughputChart;
        
        // Data storage
        const historyData = {
            timestamps: [],
            loss: [],
            tokensPerSecond: []
        };
        
        const workers = {};
        
        // Connect to WebSocket server
        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.hostname;
            const wsPort = 8765; // Change this if your server uses a different port
            
            const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}`;
            
            document.getElementById('connection-status').textContent = 'Connecting...';
            document.getElementById('reconnect-btn').style.display = 'none';
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                reconnectAttempts = 0;
                
                // Request current stats
                socket.send(JSON.stringify({
                    type: 'get_stats',
                    stats_type: 'all'
                }));
            };
            
            socket.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('reconnect-btn').style.display = 'inline-block';
                
                // Auto-reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, reconnectInterval);
                }
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
            
            socket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
        }
        
        // Handle incoming messages
        function handleMessage(data) {
            const messageType = data.type;
            
            switch (messageType) {
                case 'server_info':
                    // Update server info
                    break;
                    
                case 'worker_stats':
                case 'worker_stats_update':
                    updateWorkerStats(data.worker_id, data.stats);
                    break;
                    
                case 'aggregated_stats':
                case 'aggregated_stats_update':
                    updateAggregatedStats(data.stats);
                    break;
                    
                case 'stats_history':
                    updateStatsHistory(data.history);
                    break;
                    
                case 'worker_heartbeat':
                    updateWorkerStatus(data.worker_id, data.status);
                    break;
                    
                default:
                    console.log('Unknown message type:', messageType);
            }
        }
        
        // Update aggregated stats
        function updateAggregatedStats(stats) {
            document.getElementById('current-loss').textContent = 
                stats.loss ? stats.loss.toFixed(4) : '-';
            
            document.getElementById('current-step').textContent = 
                stats.steps || '-';
            
            document.getElementById('tokens-processed').textContent = 
                stats.tokens ? formatNumber(stats.tokens) : '-';
            
            document.getElementById('tokens-per-second').textContent = 
                stats.tokens_per_second ? formatNumber(stats.tokens_per_second.toFixed(2)) : '-';
            
            document.getElementById('active-workers').textContent = 
                stats.workers || '-';
            
            // Calculate runtime
            if (stats.start_time) {
                const runtime = Math.floor((Date.now() / 1000) - stats.start_time);
                document.getElementById('runtime').textContent = formatTime(runtime);
            }
            
            // Add to history data
            if (stats.loss && stats.tokens_per_second) {
                const timestamp = stats.last_update || Date.now() / 1000;
                
                historyData.timestamps.push(new Date(timestamp * 1000));
                historyData.loss.push(stats.loss);
                historyData.tokensPerSecond.push(stats.tokens_per_second);
                
                // Keep only the last 100 points
                if (historyData.timestamps.length > 100) {
                    historyData.timestamps.shift();
                    historyData.loss.shift();
                    historyData.tokensPerSecond.shift();
                }
                
                updateCharts();
            }
        }
        
        // Update worker stats
        function updateWorkerStats(workerId, stats) {
            // Create or update worker in our tracking
            if (!workers[workerId]) {
                workers[workerId] = {
                    id: workerId,
                    status: 'unknown',
                    stats: {}
                };
                
                // Create worker card
                createWorkerCard(workerId);
            }
            
            // Update worker data
            workers[workerId].stats = stats;
            
            // Update worker card
            updateWorkerCard(workerId);
        }
        
        // Update worker status
        function updateWorkerStatus(workerId, status) {
            if (!workers[workerId]) {
                workers[workerId] = {
                    id: workerId,
                    status: status,
                    stats: {}
                };
                
                // Create worker card
                createWorkerCard(workerId);
            } else {
                workers[workerId].status = status;
                
                // Update status indicator
                const statusIndicator = document.querySelector(`.worker-card[data-worker-id="${workerId}"] .status-indicator`);
                if (statusIndicator) {
                    statusIndicator.className = `status-indicator status-${status}`;
                }
            }
        }
        
        // Update stats history
        function updateStatsHistory(history) {
            // Clear existing history
            historyData.timestamps = [];
            historyData.loss = [];
            historyData.tokensPerSecond = [];
            
            // Process history entries
            history.forEach(entry => {
                if (entry.type === 'aggregated' && entry.stats) {
                    historyData.timestamps.push(new Date(entry.timestamp * 1000));
                    historyData.loss.push(entry.stats.loss || 0);
                    historyData.tokensPerSecond.push(entry.stats.tokens_per_second || 0);
                }
            });
            
            // Sort by timestamp
            const indices = Array.from(historyData.timestamps.keys())
                .sort((a, b) => historyData.timestamps[a] - historyData.timestamps[b]);
            
            historyData.timestamps = indices.map(i => historyData.timestamps[i]);
            historyData.loss = indices.map(i => historyData.loss[i]);
            historyData.tokensPerSecond = indices.map(i => historyData.tokensPerSecond[i]);
            
            updateCharts();
        }
        
        // Create a new worker card
        function createWorkerCard(workerId) {
            const workerContainer = document.getElementById('worker-container');
            
            const card = document.createElement('div');
            card.className = 'worker-card';
            card.setAttribute('data-worker-id', workerId);
            
            const header = document.createElement('div');
            header.className = 'worker-header';
            
            const statusIndicator = document.createElement('span');
            statusIndicator.className = 'status-indicator status-unknown';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'worker-name';
            nameSpan.textContent = workerId;
            
            header.appendChild(statusIndicator);
            header.appendChild(nameSpan);
            
            const statsDiv = document.createElement('div');
            statsDiv.className = 'worker-stats';
            
            card.appendChild(header);
            card.appendChild(statsDiv);
            
            workerContainer.appendChild(card);
        }
        
        // Update a worker card with latest stats
        function updateWorkerCard(workerId) {
            const worker = workers[workerId];
            if (!worker) return;
            
            const card = document.querySelector(`.worker-card[data-worker-id="${workerId}"]`);
            if (!card) return;
            
            const statsDiv = card.querySelector('.worker-stats');
            if (!statsDiv) return;
            
            // Update status indicator
            const statusIndicator = card.querySelector('.status-indicator');
            if (statusIndicator) {
                statusIndicator.className = `status-indicator status-${worker.status || 'unknown'}`;
            }
            
            // Build stats HTML
            let statsHtml = '';
            
            if (worker.stats.device_type) {
                statsHtml += `<div><strong>Type:</strong> ${worker.stats.device_type}</div>`;
            }
            
            if (worker.stats.loss) {
                statsHtml += `<div><strong>Loss:</strong> ${worker.stats.loss.toFixed(4)}</div>`;
            }
            
            if (worker.stats.tokens) {
                statsHtml += `<div><strong>Tokens:</strong> ${formatNumber(worker.stats.tokens)}</div>`;
            }
            
            if (worker.stats.steps) {
                statsHtml += `<div><strong>Steps:</strong> ${worker.stats.steps}</div>`;
            }
            
            if (worker.stats.gpu_util) {
                statsHtml += `<div><strong>GPU Util:</strong> ${worker.stats.gpu_util}%</div>`;
            }
            
            if (worker.stats.memory_used) {
                statsHtml += `<div><strong>Memory:</strong> ${formatMemory(worker.stats.memory_used)}</div>`;
            }
            
            if (worker.stats.last_update) {
                const lastUpdateTime = new Date(worker.stats.last_update * 1000);
                statsHtml += `<div><strong>Last Update:</strong> ${lastUpdateTime.toLocaleTimeString()}</div>`;
            }
            
            statsDiv.innerHTML = statsHtml;
        }
        
        // Initialize charts
        function initCharts() {
            const lossCtx = document.getElementById('loss-chart').getContext('2d');
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute'
                            },
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        }
                    }
                }
            });
            
            const throughputCtx = document.getElementById('throughput-chart').getContext('2d');
            throughputChart = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Tokens/second',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute'
                            },
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Tokens/second'
                            }
                        }
                    }
                }
            });
        }
        
        // Update charts with latest data
        function updateCharts() {
            if (!lossChart || !throughputChart) return;
            
            // Update loss chart
            lossChart.data.labels = historyData.timestamps;
            lossChart.data.datasets[0].data = historyData.loss;
            lossChart.update();
            
            // Update throughput chart
            throughputChart.data.labels = historyData.timestamps;
            throughputChart.data.datasets[0].data = historyData.tokensPerSecond;
            throughputChart.update();
        }
        
        // Helper function to format numbers with commas
        function formatNumber(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        }
        
        // Helper function to format memory
        function formatMemory(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
            if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
            return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
        }
        
        // Helper function to format time as HH:MM:SS
        function formatTime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            
            return [
                hours.toString().padStart(2, '0'),
                minutes.toString().padStart(2, '0'),
                secs.toString().padStart(2, '0')
            ].join(':');
        }
        
        // Initialize the dashboard
        function init() {
            // Initialize charts
            initCharts();
            
            // Connect to WebSocket
            connectWebSocket();
            
            // Setup reconnect button
            document.getElementById('reconnect-btn').addEventListener('click', connectWebSocket);
            
            // Setup periodic updates for runtime
            setInterval(() => {
                const runtimeElement = document.getElementById('runtime');
                if (runtimeElement.textContent !== '-') {
                    const parts = runtimeElement.textContent.split(':');
                    if (parts.length === 3) {
                        const hours = parseInt(parts[0]);
                        const minutes = parseInt(parts[1]);
                        const seconds = parseInt(parts[2]);
                        
                        let totalSeconds = hours * 3600 + minutes * 60 + seconds + 1;
                        runtimeElement.textContent = formatTime(totalSeconds);
                    }
                }
            }, 1000);
        }
        
        // Start the dashboard when the page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""