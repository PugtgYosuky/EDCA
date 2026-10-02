# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app


# Create conda environment
RUN apt-get update \
&& apt-get install -y --no-install-recommends libgomp1 \
&& rm -rf /var/lib/apt/lists/*

# Copy environment.yml first (so caching works properly)
COPY environment.yml .
RUN conda env create -f environment.yml \
 && conda clean -afy


# Ensure the environment is activated by default in RUN commands
SHELL ["conda", "run", "-n", "edca-v1", "/bin/bash", "-c"]

# # Copy EDCA source code
# COPY edca/edca/ ./edca

# # Ensure conda environment is used when running the container
# CMD ["conda", "run", "--no-capture-output", "-n", "edca-lab", "python", "introduction-to-edca.py"]
