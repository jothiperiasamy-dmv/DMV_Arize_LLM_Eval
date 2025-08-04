import os
import json
import time
from typing import List
from opentelemetry import trace
from opentelemetry.trace import Span
from arize.otel import register
from openinference.semconv.trace import SpanAttributes, MessageAttributes

def lambda_handler(event, context):
    print("🔧 Lambda function started")

    # Load credentials from environment variables
    ARIZE_SPACE_ID = os.environ.get("ARIZE_SPACE_ID")
    ARIZE_API_KEY = os.environ.get("ARIZE_API_KEY")
    OTEL_CERTIFICATE_PATH = os.environ.get("OTEL_CERTIFICATE_PATH", "cacert.pem")

    if not ARIZE_SPACE_ID or not ARIZE_API_KEY:
        print("❌ Missing ARIZE_SPACE_ID or ARIZE_API_KEY environment variables")
        raise ValueError("Please set ARIZE_SPACE_ID and ARIZE_API_KEY as environment variables.")

    # Set certificate path for OTEL exporter
    os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = OTEL_CERTIFICATE_PATH
    print(f"🔐 Certificate set from: {OTEL_CERTIFICATE_PATH}")

    # Register the tracer
    print("📡 Registering tracer provider...")
    tracer_provider = register(
        endpoint="https://llm-dev.dmv.ca.gov/v1/traces",
        space_id=ARIZE_SPACE_ID,
        api_key=ARIZE_API_KEY,
        project_name=os.environ["ARIZE_PROJECT_NAME"]
    )

    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)
    print("✅ Tracer registered successfully")

    # Get messages from event or use default
    example_messages = event.get("messages", [
        {"role": "user", "content": "Am I eligible to renew my registration online?"},
        {"role": "assistant", "content": """You can renew your registration online if:
● You have access to the Internet.
● You have a valid credit card, debit card, or checking account.
● You know the last 5 digits of the Vehicle Identification Number (VIN) or
the Hull Identification Number (HIN) for your vehicle or vessel.
● You have insurance for your vehicle OR you are registering a vehicle
that does not need insurance (like a trailer).
● DMV has electronic smog certification information for your vehicle on
file.
You cannot renew your registration online if you do not:
● Have a credit card, debit card, or checking account.
● Know your vehicle’s VIN or HIN.
Vehicle registration fees can be paid online, however registration is not
complete if the vehicle requires a smog certificate or proof of insurance."""}
    ])
    print(f"📥 Loaded {len(example_messages)} message(s) for tracing")

    # Function to attach OpenInference attributes
    def set_input_attrs(span: Span, messages: List[dict], prompt_template: str, prompt_vars: dict | str) -> None:
        if not messages:
            print("⚠️ No messages provided to trace")
            return

        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            messages[-1].get("content", "")
        )

        for idx, msg in enumerate(messages):
            span.set_attribute(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}",
                msg.get("role", "unknown")
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                msg.get("content", "")
            )
            print(f"📝 Set span attributes for message {idx}: {msg['role']}")

        if prompt_template:
            span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, prompt_template)
            print("📄 Prompt template set")

        if prompt_vars:
            span.set_attribute(
                SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
                json.dumps(prompt_vars)
            )
            print("🔧 Prompt variables set")

    # Create a trace span and set attributes
    print("🚀 Starting span: llm-trace-example")
    with tracer.start_as_current_span("llm-trace-example") as span:
        set_input_attrs(
            span,
            messages=example_messages,
            prompt_template="What's the weather like in {city} today?",
            prompt_vars={"city": "Sacramento"}
        )
        print("✅ Span completed and attributes set")
        time.sleep(1)  # ensure span is flushed before Lambda exits

    print("🏁 Lambda function finished")
    return {
        "statusCode": 200,
        "body": json.dumps("Trace sent successfully.")
    }























# import logging
# import json
# # import pandas as pd
# import os
# # from arize.pandas.logger import Client,Schema
# # from arize.utils.types import Environments, ModelTypes
# # from phoenix.evals import llm_classify, HALLUCINATION_PROMPT_TEMPLATE, HALLUCINATION_PROMPT_RAILS_MAP,OpenAIModel,BedrockModel

# from arize.otel import register,Transport
# from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
# from opentelemetry.trace import Status, StatusCode

# ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
# ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
# ARIZE_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME")
# ARIZE_END_POINT = os.getenv("ARIZE_END_POINT")

# # llm_model = OpenAIModel(
# #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# #     model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
# #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
# #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# # )

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# def lambda_handler(event, context):
#     logger.info(f"Lambda handler started with event: {event}")
#     try:
#         if "body" in event:
#             body = event["body"]
#             if event.get("isBase64Encoded"):
#                 body = base64.b64decode(body).decode("utf-8")
#             req_body = json.loads(body)
#         else:
#             req_body = event  # Use event directly if no "body"

#         logger.info(f"Request body: {req_body}")

#         prompt = req_body.get("prompt")
#         response_txt = req_body.get("response")
#         retrieved_content = json.loads(req_body.get("retrieved_content"))



#         if not all([prompt, response_txt, retrieved_content]):
#             return _response(400, {"error": "Missing required field(s)"})

#         tracer_provider = register(
#             space_id=ARIZE_SPACE_KEY,
#             api_key=ARIZE_API_KEY,
#             project_name=ARIZE_PROJECT_NAME,
#             endpoint=ARIZE_END_POINT,
#             transport = Transport.HTTP
#         )

#         tracer = tracer_provider.get_tracer(__name__)

#         with tracer.start_as_current_span("basic-llm-span") as span:
#             # span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
#             span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
#             span.set_attribute(SpanAttributes.OUTPUT_VALUE, response_txt)
#             span.set_attribute(SpanAttributes.LLM_MODEL_NAME, os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
#             span.set_status(Status(StatusCode.OK))

#             with tracer.start_as_current_span("Retrieved Content") as vector_search_span:
#                 for i, doc in enumerate(retrieved_content):
#                     vector_search_span.set_attribute(f"retrieval.documents.{i}.document.id", i)
#                     vector_search_span.set_attribute(f"retrieval.documents.{i}.document.content", doc["content"])

#         # trace_id = format(span.get_span_context().trace_id, '032x')
#         # span_id = format(span.get_span_context().span_id, '016x')

#         # logger.info(f"Trace ID: {trace_id}, Span ID: {span_id}")

#         # eval_result = run_eval(prompt, response_txt, retrieved_content)
#         # log_to_arize(prompt, response_txt, retrieved_content, eval_result, trace_id, span_id)

#         # return _response(200, {"hallucination_eval": eval_result})
#         return _response(200,"Success")

#     except Exception as e:
#         # logger.exception("Error processing request")
#         return _response(500, {"error": str(e)})

# def _response(status_code,message):
#     return {
#         "statusCode":   status_code,
#         "Message": message
#     }
#     # return {
#     #     "statusCode": status_code,
#     #     "headers": {"Content-Type": "application/json"},
#     #     "body": json.dumps(body)
#     # }


# def run_eval(prompt, response, retrieved_content):
#     # Prepare dataframe for Arize eval
#     df = pd.DataFrame([{
#         "input": prompt,
#         "output": response,
#         "reference": retrieved_content
#     }])

#     # Hallucination evaluation
#     eval_result = llm_classify(
#         dataframe=df,
#         template=HALLUCINATION_PROMPT_TEMPLATE,
#         model=llm_model,  # Use your desired eval model
#         rails=list(HALLUCINATION_PROMPT_RAILS_MAP.values()),
#         provide_explanation=True,
#     )
#     # Return hallucination result from first row
#     return eval_result.iloc[0].to_dict()

# def log_to_arize(prompt, response, retrieved_content, hallucination_result,trace_id, span_id):
#     try:
#         arize_client = Client(space_id=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

#         hallucination_score = 1 if hallucination_result .get("label") == "hallucinated" else 0.0

#         log_df = pd.DataFrame({
#         "context.span_id": [span_id],
#         "input": [prompt],
#         "output": [response],
#         "eval.hallucination.label": [hallucination_result.get("label", None)],  # replace with your eval result
#         "eval.hallucination.score":[hallucination_score] ,
#         "eval.hallucination.explanation": [hallucination_result.get("explanation", None)]
#     })

        
        
#     # Log evaluation
#         resp = arize_client.log_evaluations_sync(
#             dataframe=log_df,
#             model_id=ARIZE_PROJECT_NAME,
#         )

#         logging.info("Evaluation logged to Arize.")

#     except BaseException as e:
#         logging.error(f"Failed to log evaluation to Arize: {str(e)}")
#         raise e