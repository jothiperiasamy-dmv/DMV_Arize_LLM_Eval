from arize.exporter import ArizeExportClient
from arize.utils.types import Environments
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = r"D:\Users\mwjmp26\cert.pem"

client = ArizeExportClient(api_key=os.getenv("ARIZE_API_KEY"))

primary_df = client.export_model_to_df(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    model_id="aws-bedrock-agent-tracing-auto-streamlit app",
    environment=Environments.TRACING,
    start_time=datetime.now(timezone.utc) - timedelta(days=1),
    end_time=datetime.now(timezone.utc),
)

print(primary_df.head())