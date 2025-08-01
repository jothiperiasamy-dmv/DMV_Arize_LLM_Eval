import azure.functions as func
import json
import logging



import azure.functions as func
from langdetect import detect
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import time
import os

# from wrapt_timeout_decorator import *



import logging
import azure.functions as func
import json
import pandas as pd
import os
from arize.pandas.logger import Client,Schema
from arize.utils.types import Environments, ModelTypes
from phoenix.evals import llm_classify, HALLUCINATION_PROMPT_TEMPLATE, HALLUCINATION_PROMPT_RAILS_MAP,OpenAIModel,BedrockModel

from arize.otel import register
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.trace import Status, StatusCode

# https://curl.se/ca/cacert.pem
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = "cacert.pem"


try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    model = None
    logging.error(f"Error loading SentenceTransformer model: {e}")



app = func.FunctionApp()



# Configure your Arize credentials
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

# llm_model = BedrockModel(model_id=)


@app.function_name(name="AG_Arize_Function")
@app.route(route="AG_Arize_Function")
def AG_Arize_Function(req: func.HttpRequest) -> func.HttpResponse:

    try:

        req_body = req.get_json()
        prompt = req_body.get("prompt")
        response = req_body.get("response")
        retrieved_content = req_body.get("retrieved_content")

        if not all([prompt, response, retrieved_content]):
            return func.HttpResponse(
                json.dumps({"error": "Missing required field(s)"}),
                status_code=400, mimetype="application/json"
            )

        # 1. Setup
        tracer_provider = register(
            space_id= ARIZE_SPACE_KEY,    
        api_key=ARIZE_API_KEY,
        project_name= ARIZE_PROJECT_NAME,
        # endpoint=ARIZE_END_POINT
        )

        tracer = tracer_provider.get_tracer(__name__)

        # 2. Create manual span
        with tracer.start_as_current_span("basic-llm-span") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE,prompt)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
            span.set_status(Status(StatusCode.OK))
            

            with tracer.start_as_current_span("Retrieved Content") as vector_search_span:
                for i, document in enumerate(retrieved_content):
                    vector_search_span.set_attribute(f"retrieval.documents.{i}.document.id", i)
                    # vector_search_span.set_attribute(f"retrieval.documents.{i}.document.score", document["document.score"])
                    vector_search_span.set_attribute(f"retrieval.documents.{i}.document.content", document["content"])

        trace_id = format(span.get_span_context().trace_id, '032x')
        span_id = format(span.get_span_context().span_id, '016x')
        logging.info(f"Trace ID: {trace_id}, Span ID: {span_id}")


        eval_result = run_eval(prompt, response, retrieved_content)
        log_to_arize(prompt, response, retrieved_content, eval_result, trace_id, span_id)

        return func.HttpResponse(
            json.dumps({"hallucination_eval": eval_result}),
            status_code=200, mimetype="application/json"
        )
    except Exception as e:
        logging.error(str(e))
        logging.exception("Error processing request")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500, mimetype="application/json"
        )


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
    


# @app.function_name(name="AG_API_Transaction_Enquiry")
# @app.route(route="AG_API_Transaction_Enquiry")

# def AG_API_Transaction_Enquiry(req: func.HttpRequest) -> func.HttpResponse:
#     # This function is a placeholder for vehicle registration status checking. It's just to demonstrate how to handle api call with copilot studio agent.
#     # In a real-world scenario, you would replace this with actual logic to check vehicle registration status.
#     try:
#         start_time = time.time()
#         TIMEOUT_SECS = int(os.environ["TIMEOUT_SECS"])
#         logging.info('Python HTTP trigger function processed a request.')
#         plate = req.params.get('plate')
#         if not plate:
#             try:
#                 req_body = req.get_json()
#             except ValueError:
#                 pass
#             else:
#                 plate = req_body.get('plate')
        
#         if plate and time.time() - start_time < TIMEOUT_SECS:
#             result = {
#                 "License Plate": plate,
#                 "Status": "Active",
#                 "Expiry Date": "12-25-2025",
#                 "message" : ""
#             }
#             return func.HttpResponse(
#                 json.dumps(result),
#                 mimetype="application/json",
#                 status_code=200
#             )
#         else:
#             raise TimeoutError("Request took too long to process, exceeding the 10 seconds limit.")
#     except TimeoutError as e:
#         logging.error(f"Timeout Error in AG_API_Transaction_Enquiry: {e}", exc_info=True)
#         result = {
#                 "License Plate": plate,
#                 "Status": "Active",
#                 "Expiry Date": "12-25-2025",
#                 "message": "Request timed out. Please try again later."
#             }
#         return func.HttpResponse(
#                 json.dumps(result),
#                 mimetype="application/json",
#                 status_code=200
#             )
#     except Exception as e:
#         print("error in AG_API_Transaction_Enquiry: ", str(e))
#         logging.error(f"Error in AG_API_Transaction_Enquiry: {e}", exc_info=True)
        
#         result = {
#                 "License Plate": plate,
#                 "Status": "Active",
#                 "Expiry Date": "12-25-2025",
#                 "message": "Unhandled exception in AG_API_Transaction_Enquiry: " + str(e)
#             }
#         logging.error(f"Unhandled exception in AG_API_Transaction_Enquiry: {e}", exc_info=True)
#         return func.HttpResponse(
#                 json.dumps(result),
#                 mimetype="application/json",
#                 status_code=200
#             )
    
@app.function_name(name="AG_LangDetect")
@app.route(route="AG_LangDetect")
def AG_LangDetect(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function AG_DEV_LangDetect_Azure_Function started processing a request.')
    start_time = time.time()
    TIMEOUT_SECS = int(os.environ["TIMEOUT_SECS"])

    try:
        text_input = req.params.get('text_input')
        logging.debug(f"Query parameter 'text_input': {text_input}")

        if not text_input:
            try:
                req_body = req.get_json()
                logging.debug(f"Request body JSON: {req_body}")
            except ValueError as ve:
                logging.warning(f"Failed to parse JSON body: {ve}", exc_info=True)
                req_body = None
            except Exception as e:
                logging.error(f"Unexpected error while parsing JSON body: {e}", exc_info=True)
                req_body = None
            else:
                text_input = req_body.get('text_input') if req_body else None
                logging.debug(f"'text_input' from JSON body: {text_input}")

        if text_input:
            logging.info(f"Received text_input: {text_input}")
            try:
                if time.time()-start_time<TIMEOUT_SECS:
                    language = detect(text=text_input)
                    logging.info(f"Detected language: {language}")

                    result = {
            "language":language,
                "message": ""
            }
                    return func.HttpResponse(
                json.dumps(result),
                mimetype="application/json",
                status_code=200
            )
                else:
                    raise TimeoutError("Request took too long to process, exceeding the 10 seconds limit.")
            
            except TimeoutError as e:
                logging.error(f"Timeout Error in AG_LangDetect: {e}", exc_info=True)
                result = {
                    "language":"",
                        "message": "Request timed out. Please try again later."
                    }
                return func.HttpResponse(
                        json.dumps(result),
                        mimetype="application/json",
                        status_code=200
                    )
            except Exception as e:
                logging.error(f"Error during language detection: {e}", exc_info=True)
                return func.HttpResponse(
                    "Error during language detection. Check logs for details.",
                    status_code=500
                )
        else:
            logging.info("No 'text_input' provided in query or body.")
            return func.HttpResponse(
                "This HTTP triggered function executed successfully. Pass 'text_input' in the query string or in the request body for a language detection response.",
                status_code=200
            )
    
    except Exception as e:
        logging.error(f"Unhandled exception in AG_DEV_LangDetect_Azure_Function: {e}", exc_info=True)
        return func.HttpResponse(
            "An unexpected error occurred. Check logs for details.",
            status_code=500
        )		

@app.function_name(name="AG_Generate_Embedding")
@app.route(route="AG_Generate_Embedding")
def AG_Generate_Embedding(req: func.HttpRequest) -> func.HttpResponse:
    try:
        start_time = time.time()
        TIMEOUT_SECS = int(os.environ["TIMEOUT_SECS"])

        logging.info('Python HTTP trigger function processed a request.')

        if not model:
            return func.HttpResponse(
                "SentenceTransformer model not loaded. Check function logs.",
                status_code=500
            )
        try:
            req_body = req.get_json()
        except ValueError:
            logging.error("Invalid JSON format in request body")
            return func.HttpResponse("Please pass a valid JSON object in the request body", status_code=400)

        text_to_embed = req_body.get('text')

        if text_to_embed and (time.time() - start_time < TIMEOUT_SECS):
            print("text_to_embed - ",text_to_embed)
            embedding = model.encode(text_to_embed).tolist() # .tolist() to make it JSON serializable
            return func.HttpResponse(
                json.dumps({"embedding": embedding, "text": text_to_embed,"message":""}),
                mimetype="application/json"
            )
        else:
            raise TimeoutError("Request took too long to process, exceeding the 10 seconds limit.")
                
    except TimeoutError as e:
            logging.error(f"Timeout Error in AG_Generate_Embedding: {e}", exc_info=True)
            result = {
                "embedding":[],
                "text": "",
                "message": "Request timed out. Please try again later."
                }
            return func.HttpResponse(
                json.dumps(result),
                    mimetype="application/json",
                    status_code=200
                )
    except Exception as e:
        logging.error(f"Unhandled exception in AG_Generate_Embedding: {e}", exc_info=True)
        return func.HttpResponse(
            "An unexpected error occurred. Check logs for details.",
            status_code=500
        )

@app.function_name(name="AG_Semantic_Similarity")
@app.route(route="AG_Semantic_Similarity")
def AG_Semantic_Similarity(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function AG_DEV_Semantic_Similarity processed a request.')
    start_time = time.time()
    TIMEOUT_SECS = int(os.environ["TIMEOUT_SECS"])

    # Initialize default response values
    # These will be part of the response regardless of match status
    response_match_found = False
    response_similarity_question = None # Use None for JSON null
    response_similarity_score = None    # Use None for JSON null
    response_message = "Processing..." # Default message

    try:
        req_body = req.get_json()
    except ValueError:
        logging.error("Invalid JSON format in request body")
        response_message = "Invalid JSON format in request body"
        payload = {
            "match_found": response_match_found,
            "similarity_question": response_similarity_question,
            "similarity_score": response_similarity_score,
            "message": response_message
        }
        return func.HttpResponse(json.dumps(payload), status_code=400, mimetype="application/json")
    
    query_embedding_list = req_body.get('query_embedding')
    candidate_embeddings_data = req_body.get('candidate_embeddings') 
    similarity_threshold = float(req_body.get('similarity_threshold'))

    if not query_embedding_list or not isinstance(query_embedding_list, list):
        response_message = "Please pass 'query_embedding' as a list in the JSON body"
        payload = {
            "match_found": response_match_found,
            "similarity_question": response_similarity_question,
            "similarity_score": response_similarity_score,
            "message": response_message
        }
        return func.HttpResponse(json.dumps(payload), status_code=400, mimetype="application/json")

    if not candidate_embeddings_data or not isinstance(candidate_embeddings_data, list):
        response_message = "Please pass 'candidate_embeddings' as a list of objects in the JSON body"
        logging.error(f"'candidate_embeddings' is missing or not a list. Received: {candidate_embeddings_data}")
        payload = {
            "match_found": response_match_found,
            "similarity_question": response_similarity_question,
            "similarity_score": response_similarity_score,
            "message": response_message
        }
        return func.HttpResponse(json.dumps(payload), status_code=400, mimetype="application/json")
    
    try:
        query_vec = np.array(query_embedding_list).reshape(1, -1)
        
        best_match_details_for_threshold = None # Stores the best match that meets the threshold
        highest_similarity_score_overall = -1.0 # Stores the absolute highest similarity found

        if not candidate_embeddings_data:
            response_message = "No candidates provided to compare."
            logging.info(response_message)
        else:
            for candidate in candidate_embeddings_data:
                embedding_vector_str = candidate.get('embedding_vector') 
                actual_question = candidate.get('actual_question_text')

                if not embedding_vector_str or not isinstance(embedding_vector_str, str):
                    logging.warning(f"Skipping candidate with question '{actual_question}' due to missing or invalid 'embedding_vector' string.")
                    continue
                
                try:
                    candidate_vec_list = json.loads(embedding_vector_str)
                    if not isinstance(candidate_vec_list, list):
                        logging.warning(f"Parsed 'embedding_vector' for question '{actual_question}' is not a list.")
                        continue
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse 'embedding_vector' string for question '{actual_question}'. String was: {embedding_vector_str}")
                    continue

                try:
                    candidate_db_vec = np.array(candidate_vec_list).reshape(1, -1)
                    sim = cosine_similarity(query_vec, candidate_db_vec)[0][0]
                    
                    # Update the overall highest similarity found so far
                    if sim > highest_similarity_score_overall:
                        highest_similarity_score_overall = sim

                    # Check if this candidate is the best one *that also meets the threshold*
                    if sim >= similarity_threshold:
                        if best_match_details_for_threshold is None or sim > best_match_details_for_threshold["similarity_score"]:
                            best_match_details_for_threshold = {
                                "question": actual_question,
                                "similarity_score": float(sim)
                            }
                except Exception as e:
                    logging.error(f"Error calculating similarity for question '{actual_question}': {e}")
                    continue
            
            # After checking all candidates, determine the final response values
            if best_match_details_for_threshold:
                response_match_found = True
                response_similarity_question = best_match_details_for_threshold["question"]
                response_similarity_score = best_match_details_for_threshold["similarity_score"]
                response_message = f"Best match found with similarity >= {similarity_threshold}."
                logging.info(f"Best match meeting threshold: Question '{response_similarity_question}', Score: {response_similarity_score:.4f}")
            else:
                # No match met the threshold
                response_match_found = False
                highest_score_str = f"{highest_similarity_score_overall:.4f}" if highest_similarity_score_overall > -1.0 else "N/A"
                response_message = f"No match found with similarity >= {similarity_threshold}. Highest score overall was: {highest_score_str}"
                logging.info(response_message)

        # Construct the final payload
        if time.time() - start_time < TIMEOUT_SECS:
            payload = {
                "match_found": response_match_found,
                "similarity_question": response_similarity_question,
                "similarity_score": response_similarity_score,
                "message": response_message
            }
            return func.HttpResponse(json.dumps(payload), mimetype="application/json")
        else:
            raise TimeoutError("Request took too long to process, exceeding the 10 seconds limit.")
                
    except TimeoutError as e:
            logging.error(f"Timeout Error in AG_Semantic_Similarity: {e}", exc_info=True)
            result = {
                "match_found":"",
                "similarity_question": "",
                "similarity_score": "",
                "message": "Request timed out. Please try again later."
                }
            return func.HttpResponse(
                json.dumps(result),
                    mimetype="application/json",
                    status_code=200
                )

    except Exception as e:
        logging.error(f"Error in semantic match processing main try-except: {e}")
        response_message = f"An unexpected error occurred during processing: {str(e)}"
        payload = {
            "match_found": response_match_found, # Will be false
            "similarity_question": response_similarity_question, # Will be None
            "similarity_score": response_similarity_score, # Will be None
            "message": response_message
        }
        return func.HttpResponse(json.dumps(payload), status_code=500, mimetype="application/json")
