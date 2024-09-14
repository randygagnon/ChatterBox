FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

# Install system utilities and Python3
RUN echo "==> Installing system utilities...." \
 && apt-get update -y \
 && apt-get install -y wget sudo python3 python3-pip python3-venv dos2unix openssh-client

# Set up Python Virtual Environment
RUN echo "==> Creating Python virtual environment..." \
 && python3 -m venv /opt/venv

# Activate the virtual environment and install Python packages
RUN echo "==> Installing Python packages inside virtual environment..." \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install langchain_experimental langchain langchain-openai langchain-community \
 && /opt/venv/bin/pip install chromadb openai tiktoken jq streamlit aiofiles aiohttp nest_asyncio

# Set the default PATH to use the virtual environment’s pip and python
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application files
COPY /chatterbox /chatterbox/
COPY /scripts /scripts/

# Convert script line endings (if necessary)
RUN echo "==> Convert script..." \
 && dos2unix /scripts/startup.sh

# Set the entry point to run the startup script
CMD ["/bin/bash", "/scripts/startup.sh"]
