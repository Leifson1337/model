# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# Using --default-timeout to prevent timeouts if network is slow
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Define environment variables if necessary (can also be passed at runtime)
# ENV PYTHONPATH /app

# Make port 80 available to the world outside this container (if running a web service)
# EXPOSE 80

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Define the command to run on container start
# This will show the help message for the CLI tool by default via entrypoint.sh.
# Specific commands can be passed when running the container, e.g.,
# docker run <image_name> python main.py load-data --config config/dev.json
ENTRYPOINT ["/app/entrypoint.sh"]
# Default command for entrypoint.sh (will be passed as "$@")
CMD ["python", "main.py", "--help"]
