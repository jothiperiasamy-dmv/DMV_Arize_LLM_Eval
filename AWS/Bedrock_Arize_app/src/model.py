import os
import time
import boto3
from dotenv import load_dotenv
from openinference.instrumentation.bedrock import BedrockInstrumentor
from arize.otel import register
import streamlit as st
import pandas as pd
from openinference.instrumentation import using_attributes
from opentelemetry.trace import get_current_span, format_span_id


import os
import certifi
from arize.exporter import ArizeExportClient
from arize.utils.types import Environments
from datetime import datetime, timedelta, timezone

def get_aws_agent_model_response(user_input):
    load_dotenv()
    # os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = r"D:\\Users\\mwjmp26\\OneDrive - California Department of Motor Vehicles\Documents\\arize_app\\app\\cert.pem"
    if "initialized" not in st.session_state:
        tracer_provider = register(
            endpoint= os.getenv("ARIZE_OTLP_ENDPOINT"),
            space_id=os.getenv("ARIZE_SPACE_ID"),
            api_key=os.getenv("ARIZE_API_KEY"),
            project_name="aws-bedrock-agent-tracing-auto-streamlit app",
        )
        BedrockInstrumentor().instrument(tracer_provider=tracer_provider)

        session = boto3.session.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("REGION_NAME")
        )

        st.session_state.client = session.client("bedrock-agent-runtime")
        st.session_state.initialized = True

    client = st.session_state.client

    AGENT_ID = os.getenv("BEDROCK_AGENT_ID")
    AGENT_ALIAS_ID = os.getenv("BEDROCK_AGENT_ALIAS")

    if "session_id" not in st.session_state:
        session_id = f"session-{int(time.time())}"
        st.session_state.session_id = session_id

    attributes = {
        "inputText": user_input,
        "agentId": AGENT_ID,
        "agentAliasId": AGENT_ALIAS_ID,
        "sessionId": st.session_state.session_id,
        "enableTrace": True
    }

    with using_attributes(session_id=st.session_state.session_id, user_id="anonymous-user"):
        response = client.invoke_agent(**attributes)

    model_response = ""
    retrieved_texts = []

    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            model_response += chunk["bytes"].decode("utf8")

        attribution = chunk.get("attribution", {})
        for citation in attribution.get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                text = ref.get("content", {}).get("text", "")
                if text:
                    retrieved_texts.append(text)

    retrieved_document = "\n\n---\n\n".join(retrieved_texts) or "No retrieved document found"

    new_row = pd.DataFrame([{
        "user_question": user_input,
        "retrieved_document": retrieved_document,
        "model_response": model_response,
    }])

    if "df" not in st.session_state:
        st.session_state.df = new_row
    else:
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    # run_rag_eval_to_arize()
    # session_eval()
    return model_response, st.session_state.df




def session_eval():


    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = r"D:\Users\mwjmp26\cert.pem"
    # os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = certifi.where()
    # os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = r"D:\Users\mwjmp26\Downloads\cert.pem"

    print("Calling Session evaluation.")
    

    # client = ArizeExportClient(api_key=os.getenv("ARIZE_API_KEY"),host="arize-flight.llm-dev.dmv.ca.gov")
    client = ArizeExportClient(api_key=os.getenv("ARIZE_API_KEY"))

    primary_df = client.export_model_to_df(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        model_id="aws-bedrock-agent-tracing-auto-streamlit app",
        environment=Environments.TRACING,
        start_time=datetime.now(timezone.utc) - timedelta(days=1),
        end_time=datetime.now(timezone.utc),
    )

    print(primary_df.head())

# from arize.pandas.logger import Client as ArizeClient
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import streamlit as st
# load_dotenv()

# def log_user_feedback(feedback_payload):

#     os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = r"D:\\Users\\mwjmp26\\OneDrive - California Department of Motor Vehicles\Documents\\arize_app\\app\\cert.pem"

#     # Setup Arize client
#     arize_client = ArizeClient(
#         uri= "https://llm-dev.dmv.ca.gov",
#         space_id=os.getenv("ARIZE_SPACE_ID"),
#         api_key=os.getenv("ARIZE_API_KEY"),
#         # developer_key=os.getenv("ARIZE_DEV_KEY")
#     )

#     span_id = st.session_state.get("last_span_id")
#     # span_id = "6cb487954e304081"  # For testing purposes, replace with actual span ID if available
#     if not span_id:
#         print("⚠️ No span ID found to annotate.")
#         return

#     score = 1 if feedback_payload["score"] == "👍" else 0
#     label = "thumbs_up" if score == 1 else "thumbs_down"
#     explanation = feedback_payload.get("text", "")

#     df = pd.DataFrame([{
#         "context.span_id": span_id,
#         "annotation.user_feedback.label": label,
#         "annotation.user_feedback.score": score,
#         "annotation.user_feedback.updated_by": "user",
#         "annotation.notes": explanation,
#     }])

#     arize_client.log_annotations(
#         dataframe=df,
#         project_name=os.getenv("ARIZE_MODEL_ID"),
#         validate=True,
#         verbose=True,
#     )
#     print(f"✅ Logged feedback for span {span_id}")


# import os
# from datetime import datetime, timedelta, timezone
# import pandas as pd
# from arize.exporter import ArizeExportClient
# from arize.utils.types import Environments

# def get_span_id_from_arize(session_id: str, input_text: str, days_lookback: int = 7) -> str:
#     """
#     Retrieve the span_id from Arize based on session_id and user input text.

#     Environment Variables Required:
#     - ARIZE_SPACE_ID
#     - ARIZE_API_KEY
#     - ARIZE_MODEL_ID

#     Args:
#         session_id (str): The session identifier (e.g. "session-123")
#         input_text (str): The text of the user input to match
#         days_lookback (int): Number of days to look back for traces

#     Returns:
#         str or None: The matched span_id, or None if no match is found
#     """
#     os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = r"D:\\Users\\mwjmp26\\OneDrive - California Department of Motor Vehicles\Documents\\arize_app\\app\\cert.pem"


#     # Load from environment variables
#     space_id = os.getenv("ARIZE_SPACE_ID")
#     api_key = os.getenv("ARIZE_API_KEY")
#     model_id = os.getenv("ARIZE_MODEL_ID")

#     if not all([space_id, api_key, model_id]):
#         raise ValueError("ARIZE_SPACE_ID, ARIZE_API_KEY, and ARIZE_MODEL_ID must be set in environment variables.")

#     client = ArizeExportClient(api_key=api_key, host="https://llm-dev.dmv.ca.gov")
#     now = datetime.now(timezone.utc)

#     # Export trace data
#     df = client.export_model_to_df(
#         space_id=space_id,
#         model_id=model_id,
#         environment=Environments.TRACING,
#         start_time=now - timedelta(days=days_lookback),
#         end_time=now
#     )

#     span_df = pd.DataFrame(df)

#     # Filter by session ID and user input
#     filtered_df = span_df[
#         (span_df["attributes.session.id"] == session_id) &
#         (span_df["attributes.inputText"].str.contains(input_text, na=False))
#     ]

#     if not filtered_df.empty:
#         print(f"Found {len(filtered_df)} matching traces for session '{session_id}' and input '{input_text}'")
#         span_id = filtered_df["context.span_id"].iloc[0]
#         print(f"Matched span_id: {span_id}")
#         st.session_state.last_span_id = span_id
#         return span_id

#     return None
