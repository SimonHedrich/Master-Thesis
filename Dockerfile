# Use the Debian Bookworm base image
FROM debian:bookworm

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default python command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# # Copy the source code from the host to the container
# COPY source/ .

# # Run the Python script
# CMD ["python", "stable_diffusion_example.py"]
CMD ["/bin/bash"]