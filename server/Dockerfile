# Use an official Python runtime as a base image
FROM python:latest

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Make port 6071 available outside this container
EXPOSE 8080/tcp
EXPOSE 6071/udp

# Run app.py when the container launches
CMD ["python", "-u", "server.py"]
