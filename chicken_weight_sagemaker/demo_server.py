#!/usr/bin/env python3
"""
Demo server for chicken weight estimation system.
"""

import sys
import os
import time
import base64
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import cv2
from flask import Flask, request, jsonify
import threading

from src.inference.stream_handler import RealTimeStreamProcessor
from src.utils.config.config_manager import ConfigManager

app = Flask(__name__)

# Global processor instance
processor = None

def create_test_frame():
    """Create a test frame with chicken-like objects."""
    # Create a realistic farm scene
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add background (barn floor)
    frame[:] = [101, 67, 33]  # Brown background
    
    # Add some chicken-like ellipses
    chickens = [
        ((200, 200), (50, 30), (200, 180, 150)),  # Chicken 1
        ((400, 300), (60, 35), (180, 160, 140)),  # Chicken 2
        ((150, 350), (45, 28), (190, 170, 145)),  # Chicken 3
    ]
    
    for (center, axes, color) in chickens:
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, -1)
        # Add some texture
        cv2.ellipse(frame, center, (axes[0]-10, axes[1]-5), 0, 0, 360, 
                   (color[0]-20, color[1]-20, color[2]-20), 2)
    
    # Add some noise for realism
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint for SageMaker."""
    return jsonify({'status': 'healthy', 'service': 'chicken-weight-estimator'}), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    """Main inference endpoint for SageMaker."""
    try:
        # Parse request
        request_data = request.get_json()
        
        if not request_data or 'stream_data' not in request_data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        stream_data = request_data['stream_data']
        
        # Extract required fields
        frame_data = stream_data.get('frame')
        camera_id = stream_data.get('camera_id', 'default')
        frame_sequence = stream_data.get('frame_sequence', 0)
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Process frame using the global processor
        result = processor.process_frame(frame_data, camera_id, frame_sequence)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo', methods=['GET'])
def demo():
    """Demo endpoint that processes a test frame."""
    try:
        # Create test frame
        test_frame = create_test_frame()
        
        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Create demo request
        demo_request = {
            "stream_data": {
                "frame": frame_data,
                "camera_id": "demo_camera",
                "frame_sequence": int(time.time()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + ".000Z",
                "parameters": {
                    "min_confidence": 0.4,
                    "max_occlusion": 0.7
                }
            }
        }
        
        # Process through the main endpoint
        with app.test_client() as client:
            response = client.post('/invocations', 
                                 data=json.dumps(demo_request),
                                 content_type='application/json')
            
            if response.status_code == 200:
                result = response.get_json()
                
                # Add demo metadata
                demo_result = {
                    "demo_info": {
                        "message": "This is a demo of the chicken weight estimation system",
                        "test_frame_size": f"{test_frame.shape[0]}x{test_frame.shape[1]}",
                        "processing_mode": "mock_models",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + ".000Z"
                    },
                    "processing_result": result
                }
                
                return jsonify(demo_result), 200
            else:
                return jsonify({"error": "Demo processing failed"}), 500
                
    except Exception as e:
        return jsonify({"error": f"Demo failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check endpoint."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S") + ".000Z",
            'service': 'chicken-weight-estimator',
            'version': '1.0.0',
            'components': {
                'processor': 'healthy' if processor else 'not_initialized',
                'config_manager': 'healthy',
                'models': 'mock_mode'
            },
            'performance': processor.get_processing_stats() if processor else {}
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S") + ".000Z"
        }), 503

@app.route('/stats', methods=['GET'])
def stats():
    """Get processing statistics."""
    try:
        if processor:
            stats = processor.get_processing_stats()
            stats['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S") + ".000Z"
            return jsonify(stats), 200
        else:
            return jsonify({'error': 'Processor not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    api_docs = {
        "service": "Chicken Weight Estimation API",
        "version": "1.0.0",
        "description": "Real-time chicken weight estimation using computer vision",
        "endpoints": {
            "GET /": "This documentation",
            "GET /ping": "Health check (SageMaker compatible)",
            "GET /health": "Detailed health status",
            "GET /stats": "Processing statistics",
            "GET /demo": "Demo with test frame",
            "POST /invocations": "Main inference endpoint (SageMaker compatible)"
        },
        "example_request": {
            "stream_data": {
                "frame": "<base64_encoded_image>",
                "camera_id": "farm_camera_01",
                "frame_sequence": 42,
                "timestamp": "2024-01-15T10:30:00.123Z",
                "parameters": {
                    "min_confidence": 0.4,
                    "max_occlusion": 0.7
                }
            }
        },
        "example_response": {
            "camera_id": "farm_camera_01",
            "timestamp": "2024-01-15T10:30:00.456Z",
            "frame_sequence": 42,
            "detections": [
                {
                    "chicken_id": "chicken_1",
                    "bbox": [150, 170, 250, 230],
                    "confidence": 0.85,
                    "weight_estimate": {
                        "value": 2.3,
                        "confidence": 0.78,
                        "error_range": "±0.4kg"
                    }
                }
            ],
            "total_chickens_detected": 1,
            "average_weight": 2.3,
            "processing_time_ms": 45.2,
            "status": "success"
        }
    }
    
    return jsonify(api_docs), 200

def initialize_processor():
    """Initialize the global processor."""
    global processor
    try:
        print("🔄 Initializing chicken weight estimation processor...")
        config_manager = ConfigManager()
        processor = RealTimeStreamProcessor(config_manager=config_manager)
        print("✅ Processor initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False

def main():
    """Main function to run the demo server."""
    print("🐔 CHICKEN WEIGHT ESTIMATION - DEMO SERVER")
    print("=" * 60)
    
    # Initialize processor
    if not initialize_processor():
        print("❌ Failed to start server due to processor initialization error")
        return 1
    
    print("\n🚀 Starting Flask demo server...")
    print("📡 Server will be available at:")
    print("   - Main API: http://localhost:8080/")
    print("   - Health check: http://localhost:8080/ping")
    print("   - Demo: http://localhost:8080/demo")
    print("   - Stats: http://localhost:8080/stats")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        # Run the Flask app
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())