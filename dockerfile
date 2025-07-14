# Use a modern slim Python base image
FROM python:3.9-slim AS builder

# Avoid pip warnings
ENV PYTHONUNBUFFERED=1

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    nano \
    libglib2.0-0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt/app

# Clone the necessary repositories
RUN git clone https://github.com/abel-research/ampscan.git \
 && git clone https://github.com/abel-research/OpenLimbTT.git

# Install ampscan and its dependencies
WORKDIR /opt/app/ampscan
RUN pip install --no-cache-dir numpy==1.24 scipy matplotlib pyqt5==5.15.9 vtk==9.2.6 scikit-learn==1.2.1 \
 && pip install --no-cache-dir .


# Install any OpenLimbTT-specific dependencies if required
WORKDIR /opt/app/OpenLimbTT
# Uncomment if OpenLimbTT uses requirements.txt
# RUN pip install --no-cache-dir

ENV PYTHONPATH="/opt/app/ampscan:/opt/app/OpenLimbTT:${PYTHONPATH:-}"

COPY entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

ENTRYPOINT ["/opt/app/entrypoint.sh"]
CMD ["python", "GenerateRandomLimbs.py"]
