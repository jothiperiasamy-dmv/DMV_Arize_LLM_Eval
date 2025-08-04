# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM public.ecr.aws/lambda/python:3.12

RUN dnf install -y gcc gcc-c++ make

COPY requirements.txt ${LAMBDA_TASK_ROOT}

COPY cacert.pem ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]