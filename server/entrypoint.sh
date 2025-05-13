#!/bin/bash

# Start CloudWatch agent in the background
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent \
  -config /workspace/cloudwatch-container-config.json &

# Start the Flower server
python server.py
