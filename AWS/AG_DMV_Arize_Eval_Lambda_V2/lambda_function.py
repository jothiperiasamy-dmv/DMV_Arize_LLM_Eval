import logging
import json
import pandas as pd
import os
from arize.pandas.logger import Client,Schema
from arize.utils.types import Environments, ModelTypes
from phoenix.evals import llm_classify, HALLUCINATION_PROMPT_TEMPLATE, HALLUCINATION_PROMPT_RAILS_MAP,OpenAIModel,BedrockModel

from arize.otel import register
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.trace import Status, StatusCode

ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME")
ARIZE_END_POINT = os.getenv("ARIZE_END_POINT")

llm_model = OpenAIModel(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        if "body" not in event:
            return _response(400, {"error": "No request body"})

        body = event["body"]
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        req_body = json.loads(body)

        prompt = req_body.get("prompt")
        response_txt = req_body.get("response")
        retrieved_content = json.loads(req_body.get("retrieved_content"))

        if not all([prompt, response_txt, retrieved_content]):
            return _response(400, {"error": "Missing required field(s)"})

        tracer_provider = register(
            space_id=ARIZE_SPACE_KEY,
            api_key=ARIZE_API_KEY,
            project_name=ARIZE_PROJECT_NAME,
            endpoint=ARIZE_END_POINT
        )

        tracer = tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("basic-llm-span") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response_txt)
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
            span.set_status(Status(StatusCode.OK))

            with tracer.start_as_current_span("Retrieved Content") as vector_search_span:
                for i, doc in enumerate(retrieved_content):
                    vector_search_span.set_attribute(f"retrieval.documents.{i}.document.id", i)
                    vector_search_span.set_attribute(f"retrieval.documents.{i}.document.content", doc["content"])

        trace_id = format(span.get_span_context().trace_id, '032x')
        span_id = format(span.get_span_context().span_id, '016x')

        logger.info(f"Trace ID: {trace_id}, Span ID: {span_id}")

        eval_result = run_eval(prompt, response_txt, retrieved_content)
        log_to_arize(prompt, response_txt, retrieved_content, eval_result, trace_id, span_id)

        return _response(200, {"hallucination_eval": eval_result})

    except Exception as e:
        logger.exception("Error processing request")
        return _response(500, {"error": str(e)})

def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }


def run_eval(prompt, response, retrieved_content):
    # Prepare dataframe for Arize eval
    df = pd.DataFrame([{
        "input": prompt,
        "output": response,
        "reference": retrieved_content
    }])

    # Hallucination evaluation
    eval_result = llm_classify(
        dataframe=df,
        template=HALLUCINATION_PROMPT_TEMPLATE,
        model=llm_model,  # Use your desired eval model
        rails=list(HALLUCINATION_PROMPT_RAILS_MAP.values()),
        provide_explanation=True,
    )
    # Return hallucination result from first row
    return eval_result.iloc[0].to_dict()

def log_to_arize(prompt, response, retrieved_content, hallucination_result,trace_id, span_id):
    try:
        arize_client = Client(space_id=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

        hallucination_score = 1 if hallucination_result .get("label") == "hallucinated" else 0.0

        log_df = pd.DataFrame({
        "context.span_id": [span_id],
        "input": [prompt],
        "output": [response],
        "eval.hallucination.label": [hallucination_result.get("label", None)],  # replace with your eval result
        "eval.hallucination.score":[hallucination_score] ,
        "eval.hallucination.explanation": [hallucination_result.get("explanation", None)]
    })

        
        
    # Log evaluation
        resp = arize_client.log_evaluations_sync(
            dataframe=log_df,
            model_id=ARIZE_PROJECT_NAME,
        )

        logging.info("Evaluation logged to Arize.")

    except BaseException as e:
        logging.error(f"Failed to log evaluation to Arize: {str(e)}")
        raise e