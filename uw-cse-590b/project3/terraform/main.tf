terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure the AWS Provider (Mock configuration for building/planning)
provider "aws" {
  region                      = "us-west-2"
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
  access_key                  = "mock_access_key"
  secret_key                  = "mock_secret_key"
}

# 1. SQS Queue for Asynchronous Order Processing
resource "aws_sqs_queue" "order_queue" {
  name                      = "event-driven-order-queue"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 345600 # 4 days
  receive_wait_time_seconds = 10     # Enable long polling
  
  # Dead Letter Queue configuration
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.order_dlq.arn
    maxReceiveCount     = 5
  })
}

resource "aws_sqs_queue" "order_dlq" {
  name = "event-driven-order-dlq"
}

# 2. DynamoDB Table for Order and Inventory State
resource "aws_dynamodb_table" "order_state_table" {
  name           = "OrdersAndInventory"
  billing_mode   = "PAY_PER_REQUEST" # Highly cost-effective for unpredictable spikes
  hash_key       = "PK"
  range_key      = "SK"

  attribute {
    name = "PK"
    type = "S"
  }

  attribute {
    name = "SK"
    type = "S"
  }

  # Enable Point-in-Time Recovery (PITR) for Reliability/Resilience
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Environment = "Production"
    Project     = "Project3-EventDriven"
  }
}
