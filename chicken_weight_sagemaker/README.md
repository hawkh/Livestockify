# Chicken Weight Estimation SageMaker Deployment

A real-time chicken detection and weight estimation system deployed on AWS SageMaker, optimized for live poultry farm footage with occlusion handling and distance compensation.

## Features

- Real-time processing of live poultry farm video streams
- Occlusion-robust YOLO detection (handles 50-70% occlusions)
- Distance-adaptive weight estimation neural network
- Multi-object tracking for chicken identification
- ±20-30% weight estimation error tolerance
- Camera distance compensation (2-10 meters)
- SageMaker deployment with auto-scaling

## Project Structure

```
chicken_weight_sagemaker/
├── src/
│   ├── models/
│   │   ├── detection/          # YOLO detection models
│   │   ├── weight_estimation/  # Weight estimation neural networks
│   │   └── tracking/          # Multi-object tracking
│   ├── inference/
│   │   ├── handlers/          # SageMaker inference handlers
│   │   ├── preprocessing/     # Image/frame preprocessing
│   │   └── postprocessing/    # Result formatting
│   ├── utils/
│   │   ├── camera/           # Camera calibration utilities
│   │   ├── distance/         # Distance compensation
│   │   └── config/           # Configuration management
│   └── core/
│       ├── interfaces/       # Core interfaces and data models
│       └── exceptions/       # Custom exceptions
├── docker/
│   ├── Dockerfile           # SageMaker container
│   └── requirements.txt     # Python dependencies
├── deployment/
│   ├── scripts/            # Deployment automation
│   ├── configs/            # SageMaker configurations
│   └── iam/               # IAM roles and policies
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── data/             # Test data
├── model_artifacts/       # Model weights and configs
└── examples/             # Usage examples and SDK
```

## Quick Start

1. Install dependencies:
```bash
pip install -r docker/requirements.txt
```

2. Configure camera parameters:
```bash
cp src/utils/config/camera_config.yaml.example src/utils/config/camera_config.yaml
# Edit camera_config.yaml with your camera specifications
```

3. Deploy to SageMaker:
```bash
python deployment/scripts/deploy.py --config deployment/configs/production.yaml
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.5+
- AWS CLI configured
- SageMaker permissions

## License

MIT License