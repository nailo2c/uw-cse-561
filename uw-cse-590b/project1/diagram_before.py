from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import InternetGateway
from diagrams.onprem.client import Users

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.5",
    "label": "Before: Legacy Monolithic E-Commerce Architecture",
    "labelloc": "t",
    "fontname": "Helvetica-Bold",
}

with Diagram(
    "",
    filename="before_architecture",
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    users = Users("Customers")

    with Cluster("AWS Region (single AZ)"):
        igw = InternetGateway("Internet\nGateway")

        with Cluster("VPC - 10.0.0.0/16"):
            with Cluster("Public Subnet (AZ-1 only)"):
                with Cluster("EC2: m5.xlarge\n(Single Instance - No ASG)"):
                    monolith = EC2("Monolith\n─────────\nWeb Server\n(Apache/Nginx)\n─────────\nBusiness Logic\n(Order, Payment,\nInventory, User)\n─────────\nData Access\nLayer")

                db = RDS("MySQL 8.0\n(Single-AZ)\n─────────\nNo Read Replica\nBackup: 1 day\nHardcoded Creds")

    users >> Edge(label="HTTP :80\n(no HTTPS)") >> igw >> monolith
    monolith >> Edge(label="MySQL :3306") >> db
