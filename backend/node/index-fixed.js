/**
 * Node.js Risk Simulation Engine Server
 * 
 * High-performance financial risk modeling API server built with Express.js
 * Supports multiple simulation methods for portfolio risk assessment
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 5010;

// Add process error handlers to prevent crashes
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    console.error('Stack:', error.stack);
    console.error('This should not crash the server. Logging and continuing...');
    // Don't exit the process - just log the error
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    console.error('This should not crash the server. Logging and continuing...');
    // Don't exit the process - just log the error
});

// Graceful shutdown handlers
process.on('SIGTERM', () => {
    console.log('🛑 SIGTERM received, shutting down gracefully');
    server.close(() => {
        console.log('✅ Process terminated gracefully');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('🛑 SIGINT received, shutting down gracefully');
    server.close(() => {
        console.log('✅ Process terminated gracefully');
        process.exit(0);
    });
});

// Middleware
app.use(helmet());
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:3001', 'http://localhost:5173'],
    credentials: true
}));
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Import and use routes (all API routes are defined in the routes file)
const routes = require('./routes/routes');
app.use('/', routes);

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        available_endpoints: [
            'GET /health',
            'GET /api/info',
            'GET /api/docs',
            'GET /api/examples',
            'POST /simulate',
            'POST /compare',
            'POST /benchmark'
        ],
        engine: 'node'
    });
});

// Global error handler
app.use((error, req, res, next) => {
    console.error('❌ Global error handler caught:', error);
    console.error('Error stack:', error.stack);

    if (!res.headersSent) {
        res.status(500).json({
            error: 'Internal Server Error',
            message: error.message || 'An unexpected error occurred',
            engine: 'node',
            timestamp: new Date().toISOString()
        });
    }
});

// Memory monitoring
let memoryMonitorInterval;
let heartbeatInterval;

function startMemoryMonitoring() {
    if (memoryMonitorInterval) {
        clearInterval(memoryMonitorInterval);
    }

    memoryMonitorInterval = setInterval(() => {
        const usage = process.memoryUsage();
        const mbUsage = {
            rss: Math.round(usage.rss / 1024 / 1024 * 100) / 100,
            heapTotal: Math.round(usage.heapTotal / 1024 / 1024 * 100) / 100,
            heapUsed: Math.round(usage.heapUsed / 1024 / 1024 * 100) / 100,
            external: Math.round(usage.external / 1024 / 1024 * 100) / 100,
        };

        // Log memory usage every 10 minutes
        console.log(`📊 Memory Usage: RSS: ${mbUsage.rss}MB, Heap Used: ${mbUsage.heapUsed}MB/${mbUsage.heapTotal}MB, External: ${mbUsage.external}MB`);

        // Warn if memory usage is high
        if (mbUsage.heapUsed > 500) {
            console.warn(`⚠️ High memory usage detected: ${mbUsage.heapUsed}MB heap used`);
        }

        // Force garbage collection if available (start with --expose-gc flag)
        if (global.gc && mbUsage.heapUsed > 200) {
            try {
                global.gc();
                console.log('🗑️ Garbage collection triggered');
            } catch (gcError) {
                // GC not available, that's okay
            }
        }
    }, 10 * 60 * 1000); // Every 10 minutes
}

function startHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
    }

    heartbeatInterval = setInterval(() => {
        const uptime = Math.round(process.uptime());
        const memory = Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
        console.log(`💓 Server heartbeat - Uptime: ${uptime}s, Memory: ${memory}MB, PID: ${process.pid}`);
    }, 5 * 60 * 1000); // Every 5 minutes
}

// Start server with enhanced error handling
const server = app.listen(PORT, () => {
    console.log('🚀 ================================================');
    console.log(`🚀 Node.js Risk Simulation Engine Server STARTED`);
    console.log('🚀 ================================================');
    console.log(`📊 Server running on: http://localhost:${PORT}`);
    console.log(`🏥 Health check: http://localhost:${PORT}/health`);
    console.log(`📖 API documentation: http://localhost:${PORT}/api/docs`);
    console.log(`📝 Examples: http://localhost:${PORT}/api/examples`);
    console.log(`🔧 Node.js version: ${process.version}`);
    console.log(`🔧 V8 version: ${process.versions.v8}`);
    console.log(`💾 Initial memory usage: ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`);
    console.log(`🎯 Ready to process risk simulations!`);
    console.log('🚀 ================================================');

    // Start monitoring
    startMemoryMonitoring();
    startHeartbeat();
});

// Server error handling
server.on('error', (error) => {
    console.error('❌ Server error:', error);
    if (error.code === 'EADDRINUSE') {
        console.error(`💥 Port ${PORT} is already in use. Please choose a different port.`);
        console.error(`💡 Try: $env:PORT="8010"; node index-fixed.js`);
        process.exit(1);
    } else if (error.code === 'EACCES') {
        console.error(`💥 Permission denied for port ${PORT}. Try using a port > 1024.`);
        process.exit(1);
    } else {
        console.error(`💥 Server error: ${error.message}`);
    }
});

server.on('close', () => {
    console.log('🛑 Server is closing...');
    if (memoryMonitorInterval) {
        clearInterval(memoryMonitorInterval);
    }
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
    }
});

// Handle cleanup on exit
process.on('exit', (code) => {
    console.log(`💀 Process exiting with code: ${code}`);
    if (memoryMonitorInterval) {
        clearInterval(memoryMonitorInterval);
    }
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
    }
});

module.exports = app;
