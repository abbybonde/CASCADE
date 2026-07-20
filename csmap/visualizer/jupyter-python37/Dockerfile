# Use Python 3.7 base image
FROM python:3.7

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workspace

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

