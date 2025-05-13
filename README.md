
# Federated Learning for Autonomous Driving with MLOps Integration

This repository presents a **containerized Federated Learning (FL) pipeline** for training object detection models in autonomous driving applications. Combining **Flower**, **YOLOv9**, **Docker**, **AWS**, **ClearML**, and **Optuna**, this project emphasizes **scalability, privacy-preservation, automation, and real-time monitoring**.

## Project Motivation
- Autonomous vehicles need constant learning from real-world data.
- Centralized data collection poses **privacy risks, latency & scalability issues**.
- **Federated Learning (FL)** enables decentralized model updates without raw data sharing.
- **MLOps** ensures automation, reproducibility, and production-readiness in such distributed pipelines.

## Project Goals
- Build a scalable, privacy-preserving FL pipeline for self-driving models.
- Integrate MLOps tools for monitoring, automation, and deployment.
- Implement autoscaling & failure recovery mechanisms.
- Ensure model reliability via continuous performance monitoring (mAP50-95 drift detection).

## Tech Stack & Tools
| Tool         | Category             | Purpose                                                               |
|--------------|----------------------|-----------------------------------------------------------------------|
| **YOLOv9t**   | Object Detection     | Real-time object detection model used for FL training                  |
| **Optuna**   | Hyperparameter Tuning| Bayesian optimization with pruning for efficient tuning                |
| **Flower**   | Federated Learning   | Orchestrates FL between server and multiple clients                    |
| **Docker**   | Containerization     | Isolated, reproducible client/server environments                     |
| **AWS EC2**  | Infrastructure       | Cloud instances for FL server & clients (autoscaling, crash recovery)  |
| **ClearML**  | MLOps Monitoring     | Experiment tracking, metrics logging, drift detection                  |
| **CloudWatch**| Infra Monitoring    | Tracks resource usage, participation, round duration, alerts           |

## System Architecture

![MSML605 drawio](https://github.com/user-attachments/assets/5b8ca528-7d45-478b-a558-9aef48ac55e3)


## Pipeline Workflow
1. Data Preprocessing & Augmentation
2. Hyperparameter Optimization with Optuna
3. Federated Learning with Flower
4. Deployment & Automation with Docker & AWS
5. Monitoring via ClearML & CloudWatch

## Folder Structure
```plaintext
├── client/
├── server/
├── Dockerfile.client
├── Dockerfile.server
├── data/
├── models/
├── scripts/
├── clearml/
├── aws/
├── docs/
├── README.md
├── requirements.txt
└── docker-compose.yml
```


## Monitoring & MLOps
- ClearML for experiment tracking & drift detection
- CloudWatch for infra metrics & autoscaling
- Data drift protection (>10% mAP50 drop triggers rollback)

## Results
| Metric                    | Value           |
|---------------------------|-----------------|
| Best mAP50-95 (Optuna)     | 0.293           |
| Baseline mAP50-95          | 0.102           |
| Avg Client Startup Time    | ~45 seconds     |
| Avg FL Round Duration      | ~2.3 minutes    |
| Autoscaling Responsiveness | ~1 minute       |

## Limitations & Future Scope
- Limited Non-IID simulations
- Simple FedAvg aggregation
- Shallow training epochs
- AWS dependency (cloud-centric)
- Future: Edge readiness, advanced aggregation, CI/CD, dynamic data assignment

## Contributors
- Aariz Faridi
- Anisha Katiyar
- Nikita Miller
- Yatish Sikka

## References
- McMahan et al. (AISTATS 2017)
- Flower: https://flower.dev/
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Optuna: https://optuna.org/
- ClearML: https://clear.ml/
- AWS CloudWatch Docs
