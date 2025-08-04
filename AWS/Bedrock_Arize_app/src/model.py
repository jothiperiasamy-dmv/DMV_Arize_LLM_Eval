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

def get_aws_agent_model_response(user_input):
    load_dotenv()
    os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = r"D:\\Users\\mwjmp26\\OneDrive - California Department of Motor Vehicles\Documents\\arize_app\\app\\cert.pem"
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
    return model_response, st.session_state.df