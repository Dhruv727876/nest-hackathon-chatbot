FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
