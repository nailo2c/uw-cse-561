from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ALB, CloudFront
from diagrams.aws.storage import S3
from diagrams.aws.security import SecretsManager
from diagrams.aws.management import Cloudwatch
from diagrams.aws.integration import SQS
from diagrams.onprem.client import Users

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.5",
    "label": "After: Cloud-Native E-Commerce Architecture",
    "labelloc": "t",
    "fontname": "Helvetica-Bold",
}

with Diagram(
    "",
    filename="after_architecture",
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    users = Users("Customers")

    with Cluster("AWS Region (Multi-AZ)"):
        cdn = CloudFront("CloudFront\n(CDN + TLS)")
        s3 = S3("S3\nStatic Assets\n(images, CSS, JS)")

        with Cluster("VPC - 10.0.0.0/16"):
            alb = ALB("ALB\n(HTTPS termination\n+ health checks)")

            with Cluster("Public Subnets (Multi-AZ)", graph_attr={"labeljust": "r"}):
                ecs = ECS("ECS Fargate\n(Auto Scaling)\n─────────\nMulti-AZ\nAZ-1 + AZ-2")
                secrets = SecretsManager("Secrets\nManager")
                cw = Cloudwatch("CloudWatch\nMetrics, Alarms\nDashboards")

            with Cluster("Private Subnets", graph_attr={"labeljust": "r"}):
                cache = ElastiCache("ElastiCache\n(Redis)\nProduct Catalog\nCache")
                queue = SQS("SQS\nOrder Processing\nQueue")

                with Cluster("Data Layer (Multi-AZ)"):
                    db_primary = RDS("RDS MySQL\nPrimary\n(AZ-1)")
                    db_standby = RDS("RDS MySQL\nStandby\n(AZ-2)")

    # Edge
    users >> Edge(label="HTTPS") >> cdn
    cdn >> Edge(label="static") >> s3
    cdn >> Edge(label="dynamic") >> alb

    # Compute
    alb >> ecs

    # Cache + DB
    ecs >> cache
    cache >> db_primary
    db_primary >> Edge(label="sync\nreplication") >> db_standby

    # SQS
    ecs >> Edge(xlabel="publish / consume order") >> queue

    # Security & observability
    ecs >> Edge(style="dashed", label="creds") >> secrets
    ecs >> Edge(style="dashed") >> cw
