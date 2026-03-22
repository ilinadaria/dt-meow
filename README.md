## Project Overview

- Autonomous driving system built for Duckietown  
- Uses **YOLOv5** for detecting traffic signs and ducks  
- Adds a custom **CVaR-inspired temporal filter** to clean up noisy detections  
- Focuses on staying reliable in messy, ambiguous situations
- Helps the robot make safer decisions, especially at intersections

## Repository Structure

- `assets/` – project assets and resources  
  - `assets/nn_models/` – custom-trained YOLOv5 models for detecting traffic signs and ducks

- `packages/` – main system implementation (perception, filtering, control)  
- `launchers/` – scripts for running nodes and experiments  

- `Dockerfile` – container setup for a reproducible environment  
- `dependencies-apt.txt`, `dependencies-py3.txt` – system and Python dependencies  

## How to Run

### Build

Build the project locally:

```bash
dts devel build
```

Build directly on a Duckiebot:

```bash
dts devel build -H <bot-name>
```

### Run

Run the system on the Duckiebot:

```bash
dts devel run -H <bot-name>
```