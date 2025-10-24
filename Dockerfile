# Use a slim Python 3.10 base image, as recommended in your README
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# The command to run your app
# We use --server.address=0.0.0.0 to make it accessible outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]