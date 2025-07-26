#!/usr/bin/env python3
"""
Farm Management System Integration Example

This example demonstrates how to integrate the Chicken Weight Estimation SDK
with a farm management system for continuous monitoring and data collection.
"""

import sys
from pathlib import Path
import asyncio
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import schedule
import time
import threading

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk import ChickenWeightClient, AsyncChickenWeightClient, setup_logging
from sdk.models import FarmConfig, CameraConfig, ProcessingResult
from sdk.utils import ConfigurationManager


class FarmManagementSystem:
    """
    Example farm management system integration.
    
    This class demonstrates how to:
    1. Manage multiple camera feeds
    2. Store processing results in a database
    3. Generate alerts and reports
    4. Monitor system health
    """
    
    def __init__(self, config_path: str = "farm_config.json"):
        """Initialize the farm management system."""
        self.config_manager = ConfigurationManager(config_path)
        self.client = None
        self.async_client = None
        self.db_path = "farm_data.db"
        self.farms = {}
        self.active_streams = {}
        self.alert_thresholds = {
            'min_chickens': 10,
            'max_chickens': 100,
            'min_weight': 1.0,
            'max_weight': 5.0,
            'max_processing_time': 2.0
        }
        
        # Setup logging
        setup_logging(level='INFO', log_file='farm_system.log')
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self._load_configuration()
    
    def _init_database(self):
        """Initialize SQLite database for storing results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                frame_id INTEGER,
                detection_count INTEGER,
                average_weight REAL,
                processing_time REAL,
                raw_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chicken_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER,
                weight_estimate REAL,
                confidence REAL,
                bbox_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Database initialized successfully")
    
    def _load_configuration(self):
        """Load farm and camera configuration."""
        # Initialize client
        client_config = self.config_manager.get_client_config()
        if client_config.get('endpoint_url'):
            self.client = ChickenWeightClient(**client_config)
            self.async_client = AsyncChickenWeightClient(**client_config)
            self.logger.info("Clients initialized successfully")
        
        # Load farm configurations
        farms_config = self.config_manager.get('farms', [])
        for farm_data in farms_config:
            farm = FarmConfig(
                farm_id=farm_data['farm_id'],
                name=farm_data['name'],
                location=farm_data['location'],
                cameras=[CameraConfig(**cam) for cam in farm_data['cameras']],
                contact_info=farm_data.get('contact_info')
            )
            self.farms[farm.farm_id] = farm
            self.logger.info(f"Loaded farm configuration: {farm.name}")
    
    def add_farm(self, farm_config: FarmConfig):
        """Add a new farm to the system."""
        self.farms[farm_config.farm_id] = farm_config
        self.logger.info(f"Added farm: {farm_config.name}")
    
    def start_monitoring(self, farm_id: str, camera_id: Optional[str] = None):
        """Start monitoring for a farm or specific camera."""
        if farm_id not in self.farms:
            raise ValueError(f"Farm {farm_id} not found")
        
        farm = self.farms[farm_id]
        cameras_to_monitor = [camera_id] if camera_id else [cam.camera_id for cam in farm.cameras]
        
        for cam_id in cameras_to_monitor:
            camera = farm.get_camera(cam_id)
            if not camera:
                self.logger.error(f"Camera {cam_id} not found in farm {farm_id}")
                continue
            
            # Start monitoring thread for this camera
            thread = threading.Thread(
                target=self._monitor_camera,
                args=(farm_id, camera),
                daemon=True
            )
            thread.start()
            
            self.active_streams[f"{farm_id}_{cam_id}"] = thread
            self.logger.info(f"Started monitoring: {farm.name} - {camera.name}")
    
    def _monitor_camera(self, farm_id: str, camera: CameraConfig):
        """Monitor a single camera feed."""
        stream_url = camera.rtsp_url or camera.http_url
        if not stream_url:
            self.logger.error(f"No stream URL configured for camera {camera.camera_id}")
            return
        
        def process_frame(result: ProcessingResult):
            """Process each frame result."""
            try:
                # Store result in database
                self._store_result(farm_id, camera.camera_id, result)
                
                # Check for alerts
                self._check_alerts(farm_id, camera.camera_id, result)
                
                # Log summary
                if result.detection_count > 0:
                    avg_weight = result.average_weight
                    self.logger.info(
                        f"{camera.name}: {result.detection_count} chickens" +
                        (f", avg weight: {avg_weight:.2f}kg" if avg_weight else "")
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing frame from {camera.name}: {str(e)}")
        
        try:
            # Process live stream
            self.client.process_live_stream(
                stream_url=stream_url,
                camera_id=camera.camera_id,
                callback=process_frame
            )
        except Exception as e:
            self.logger.error(f"Error monitoring camera {camera.name}: {str(e)}")
    
    def _store_result(self, farm_id: str, camera_id: str, result: ProcessingResult):
        """Store processing result in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store main result
        cursor.execute('''
            INSERT INTO processing_results 
            (timestamp, farm_id, camera_id, frame_id, detection_count, 
             average_weight, processing_time, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp,
            farm_id,
            camera_id,
            result.frame_id,
            result.detection_count,
            result.average_weight,
            result.processing_time,
            json.dumps(result.to_dict())
        ))
        
        # Store individual tracks
        for track in result.tracks:
            cursor.execute('''
                INSERT INTO chicken_tracks
                (timestamp, farm_id, camera_id, track_id, weight_estimate, 
                 confidence, bbox_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.timestamp,
                farm_id,
                camera_id,
                track.track_id,
                track.average_weight,
                track.confidence,
                json.dumps(track.bbox)
            ))
        
        conn.commit()
        conn.close()
    
    def _check_alerts(self, farm_id: str, camera_id: str, result: ProcessingResult):
        """Check for alert conditions."""
        alerts = []
        
        # Check chicken count
        if result.detection_count < self.alert_thresholds['min_chickens']:
            alerts.append({
                'type': 'LOW_CHICKEN_COUNT',
                'message': f"Low chicken count: {result.detection_count}",
                'severity': 'MEDIUM'
            })
        elif result.detection_count > self.alert_thresholds['max_chickens']:
            alerts.append({
                'type': 'HIGH_CHICKEN_COUNT',
                'message': f"High chicken count: {result.detection_count}",
                'severity': 'LOW'
            })
        
        # Check average weight
        if result.average_weight:
            if result.average_weight < self.alert_thresholds['min_weight']:
                alerts.append({
                    'type': 'LOW_AVERAGE_WEIGHT',
                    'message': f"Low average weight: {result.average_weight:.2f}kg",
                    'severity': 'HIGH'
                })
            elif result.average_weight > self.alert_thresholds['max_weight']:
                alerts.append({
                    'type': 'HIGH_AVERAGE_WEIGHT',
                    'message': f"High average weight: {result.average_weight:.2f}kg",
                    'severity': 'MEDIUM'
                })
        
        # Check processing time
        if result.processing_time > self.alert_thresholds['max_processing_time']:
            alerts.append({
                'type': 'HIGH_PROCESSING_TIME',
                'message': f"High processing time: {result.processing_time:.2f}s",
                'severity': 'LOW'
            })
        
        # Store alerts
        if alerts:
            self._store_alerts(farm_id, camera_id, alerts)
    
    def _store_alerts(self, farm_id: str, camera_id: str, alerts: List[Dict]):
        """Store alerts in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for alert in alerts:
            cursor.execute('''
                INSERT INTO alerts
                (timestamp, farm_id, camera_id, alert_type, message, severity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                farm_id,
                camera_id,
                alert['type'],
                alert['message'],
                alert['severity']
            ))
            
            self.logger.warning(f"ALERT [{alert['severity']}] {farm_id}/{camera_id}: {alert['message']}")
        
        conn.commit()
        conn.close()
    
    def generate_daily_report(self, farm_id: str, date: Optional[str] = None) -> Dict:
        """Generate daily report for a farm."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily statistics
        cursor.execute('''
            SELECT 
                camera_id,
                COUNT(*) as frame_count,
                AVG(detection_count) as avg_chickens,
                AVG(average_weight) as avg_weight,
                AVG(processing_time) as avg_processing_time,
                MIN(timestamp) as first_frame,
                MAX(timestamp) as last_frame
            FROM processing_results
            WHERE farm_id = ? AND DATE(timestamp) = ?
            GROUP BY camera_id
        ''', (farm_id, date))
        
        camera_stats = {}
        for row in cursor.fetchall():
            camera_stats[row[0]] = {
                'frame_count': row[1],
                'avg_chickens': round(row[2], 1) if row[2] else 0,
                'avg_weight': round(row[3], 2) if row[3] else 0,
                'avg_processing_time': round(row[4], 3) if row[4] else 0,
                'first_frame': row[5],
                'last_frame': row[6]
            }
        
        # Get alerts
        cursor.execute('''
            SELECT alert_type, COUNT(*) as count, severity
            FROM alerts
            WHERE farm_id = ? AND DATE(timestamp) = ?
            GROUP BY alert_type, severity
        ''', (farm_id, date))
        
        alerts_summary = {}
        for row in cursor.fetchall():
            alerts_summary[row[0]] = {
                'count': row[1],
                'severity': row[2]
            }
        
        conn.close()
        
        # Calculate totals
        total_frames = sum(stats['frame_count'] for stats in camera_stats.values())
        total_chickens = sum(stats['avg_chickens'] * stats['frame_count'] 
                           for stats in camera_stats.values())
        avg_chickens = total_chickens / total_frames if total_frames > 0 else 0
        
        report = {
            'farm_id': farm_id,
            'date': date,
            'summary': {
                'total_frames_processed': total_frames,
                'average_chickens_per_frame': round(avg_chickens, 1),
                'cameras_active': len(camera_stats),
                'total_alerts': sum(alert['count'] for alert in alerts_summary.values())
            },
            'camera_statistics': camera_stats,
            'alerts_summary': alerts_summary,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def export_data(self, farm_id: str, start_date: str, end_date: str, 
                   output_file: str = None) -> str:
        """Export data for a date range."""
        if output_file is None:
            output_file = f"farm_data_{farm_id}_{start_date}_to_{end_date}.json"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get processing results
        cursor.execute('''
            SELECT * FROM processing_results
            WHERE farm_id = ? AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (farm_id, start_date, end_date))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'farm_id': row[2],
                'camera_id': row[3],
                'frame_id': row[4],
                'detection_count': row[5],
                'average_weight': row[6],
                'processing_time': row[7],
                'raw_data': json.loads(row[8]) if row[8] else None
            })
        
        # Get tracks
        cursor.execute('''
            SELECT * FROM chicken_tracks
            WHERE farm_id = ? AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (farm_id, start_date, end_date))
        
        tracks = []
        for row in cursor.fetchall():
            tracks.append({
                'id': row[0],
                'timestamp': row[1],
                'farm_id': row[2],
                'camera_id': row[3],
                'track_id': row[4],
                'weight_estimate': row[5],
                'confidence': row[6],
                'bbox_data': json.loads(row[7]) if row[7] else None
            })
        
        conn.close()
        
        # Create export data
        export_data = {
            'farm_id': farm_id,
            'date_range': {'start': start_date, 'end': end_date},
            'export_timestamp': datetime.now().isoformat(),
            'processing_results': results,
            'chicken_tracks': tracks,
            'summary': {
                'total_results': len(results),
                'total_tracks': len(tracks),
                'date_range_days': (datetime.fromisoformat(end_date) - 
                                  datetime.fromisoformat(start_date)).days + 1
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Data exported to: {output_file}")
        return output_file
    
    def setup_scheduled_reports(self):
        """Setup scheduled daily reports."""
        def generate_all_reports():
            """Generate reports for all farms."""
            for farm_id in self.farms.keys():
                try:
                    report = self.generate_daily_report(farm_id)
                    report_file = f"daily_report_{farm_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    
                    with open(report_file, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    self.logger.info(f"Daily report generated: {report_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error generating report for {farm_id}: {str(e)}")
        
        # Schedule daily reports at 6 AM
        schedule.every().day.at("06:00").do(generate_all_reports)
        
        # Start scheduler thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("Scheduled reports setup complete")
    
    def stop_monitoring(self, farm_id: str = None, camera_id: str = None):
        """Stop monitoring for specific farm/camera or all."""
        if farm_id and camera_id:
            stream_key = f"{farm_id}_{camera_id}"
            if stream_key in self.active_streams:
                # Note: In a real implementation, you'd need a way to signal threads to stop
                self.logger.info(f"Stopping monitoring: {farm_id}/{camera_id}")
        else:
            self.logger.info("Stopping all monitoring")
            # In a real implementation, signal all threads to stop


def create_example_configuration():
    """Create an example configuration file."""
    config = {
        "client": {
            "endpoint_url": "https://your-sagemaker-endpoint.amazonaws.com/prod",
            "api_key": "your-api-key-here",
            "timeout": 30.0,
            "retry_attempts": 3,
            "retry_delay": 1.0
        },
        "farms": [
            {
                "farm_id": "sunrise_farm",
                "name": "Sunrise Poultry Farm",
                "location": "Rural County, State",
                "contact_info": {
                    "manager": "John Smith",
                    "phone": "+1-555-0123",
                    "email": "john@sunrisefarm.com"
                },
                "cameras": [
                    {
                        "camera_id": "barn_1_cam_1",
                        "name": "Barn 1 - North Camera",
                        "location": "Barn 1 North Side",
                        "rtsp_url": "rtsp://192.168.1.100:554/stream1",
                        "resolution": {"width": 1920, "height": 1080},
                        "fps": 30
                    },
                    {
                        "camera_id": "barn_1_cam_2",
                        "name": "Barn 1 - South Camera",
                        "location": "Barn 1 South Side",
                        "rtsp_url": "rtsp://192.168.1.101:554/stream1",
                        "resolution": {"width": 1920, "height": 1080},
                        "fps": 30
                    }
                ]
            },
            {
                "farm_id": "valley_ranch",
                "name": "Valley Chicken Ranch",
                "location": "Valley County, State",
                "contact_info": {
                    "manager": "Sarah Johnson",
                    "phone": "+1-555-0456",
                    "email": "sarah@valleyranch.com"
                },
                "cameras": [
                    {
                        "camera_id": "coop_a_cam_1",
                        "name": "Coop A - Main Camera",
                        "location": "Coop A Center",
                        "rtsp_url": "rtsp://192.168.2.100:554/stream1",
                        "resolution": {"width": 1280, "height": 720},
                        "fps": 25
                    }
                ]
            }
        ]
    }
    
    with open("farm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Example configuration created: farm_config.json")
    print("   Please update the endpoint URL, API key, and camera URLs")


def main():
    """Main function demonstrating farm integration."""
    print("🐔 Farm Management System Integration Example")
    print("=" * 60)
    
    # Create example configuration if it doesn't exist
    if not Path("farm_config.json").exists():
        print("📝 Creating example configuration...")
        create_example_configuration()
        print("\n⚠️  Please update the configuration file with your actual values")
        print("   Then run this script again to start monitoring")
        return
    
    try:
        # Initialize farm management system
        print("🚀 Initializing farm management system...")
        farm_system = FarmManagementSystem()
        
        # Setup scheduled reports
        farm_system.setup_scheduled_reports()
        
        # Start monitoring all farms
        print("📹 Starting monitoring for all farms...")
        for farm_id in farm_system.farms.keys():
            farm_system.start_monitoring(farm_id)
        
        print("✅ Monitoring started successfully!")
        print("\nSystem is now running. Press Ctrl+C to stop.")
        print("\nFeatures available:")
        print("- Real-time chicken detection and weight estimation")
        print("- Automatic alert generation")
        print("- Daily report generation")
        print("- Data export capabilities")
        print("- Database storage of all results")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(10)
                
                # Print status every 10 seconds
                active_streams = len(farm_system.active_streams)
                print(f"📊 Status: {active_streams} active camera streams")
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            farm_system.stop_monitoring()
            print("✅ Monitoring stopped")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.exception("Farm system error")


if __name__ == "__main__":
    main()