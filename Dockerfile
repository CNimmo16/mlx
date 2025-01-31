FROM python:3.11

RUN pip install poetry

WORKDIR /code
COPY poetry.lock pyproject.toml /code/

RUN poetry install --no-interaction --no-ansi

COPY . /code

EXPOSE 8000

CMD ["poetry", "run", "python", "hackernews/server.py"]
