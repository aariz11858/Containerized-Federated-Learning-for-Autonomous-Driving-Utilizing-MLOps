#!/bin/bash

# Start CloudWatch Agent in background
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent \
    -config /app/cloudwatch-config.json &

# Start your FL client (or server)
python client.py
