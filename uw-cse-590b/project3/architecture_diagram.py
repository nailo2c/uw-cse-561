from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import Fargate, Lambda
from diagrams.aws.network import APIGateway, NATGateway
from diagrams.aws.integration import SQS, SNS, StepFunctions
from diagrams.aws.database import Dynamodb, ElastiCache
from diagrams.aws.security import SecretsManager, KMS
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.client import Users

# The filename will be 'high_level_architecture.png'
with Diagram("High-Level Architecture: Event-Driven Order System", show=False, filename="high_level_architecture", direction="LR"):
    
    customers = Users("Customers")
    
    with Cluster("AWS Cloud (Region)"):
        cw = Cloudwatch("Observability\n(Logs/Metrics)")
        kms = KMS("Encryption at Rest")
        sm = SecretsManager("Secrets Management")

        with Cluster("VPC"):
            
            with Cluster("Public Subnet"):
                api_gw = APIGateway("API Gateway")
                nat = NATGateway("NAT Gateway")
            
            with Cluster("Private Subnet"):
                
                with Cluster("Order Intake Service"):
                    order_api = Fargate("Order API")
                
                with Cluster("Asynchronous Messaging"):
                    order_queue = SQS("Order Queue")
                    dlq = SQS("Dead Letter Queue")
                    event_bus = SNS("Event Notification")
                
                with Cluster("Fulfillment Workers"):
                    inventory_worker = Fargate("Inventory Worker")
                    notification_worker = Lambda("Notification Worker")
                    saga_orchestrator = StepFunctions("Saga Orchestrator")
                
                with Cluster("Data Stores"):
                    cache = ElastiCache("Redis Cache\n(Product Catalog)")
                    db = Dynamodb("DynamoDB\n(Orders & Inventory)")
    
    # Define connections (Data Flow)
    customers >> Edge(label="HTTPS / POST /orders") >> api_gw
    api_gw >> order_api
    
    # Read Path
    order_api >> Edge(label="Read Product/Stock") >> cache
    cache >> Edge(label="Cache Miss") >> db
    
    # Write Path (Async)
    order_api >> Edge(label="Publish Order Event\n(Return 202)") >> order_queue
    
    # Processing
    order_queue >> inventory_worker
    inventory_worker - Edge(color="red", style="dashed", label="Coordinate\nSaga") - saga_orchestrator
    inventory_worker >> Edge(label="Strong Consistency\nDeduct Stock") >> db
    
    # Notifications & DLQ
    inventory_worker >> Edge(label="Success/Fail Event") >> event_bus
    event_bus >> notification_worker
    notification_worker >> Edge(label="Send Email/SMS") >> nat # NAT is used for outbound internet access
    
    inventory_worker >> Edge(color="brown", style="dotted", label="Failed > 5 times") >> dlq
    
    # Management connections (Implicit)
    order_api >> cw
    inventory_worker >> cw
    db - kms
    order_queue - kms
    notification_worker >> sm
