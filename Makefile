# Compile production and development dependencies
compile-requirements: ensure-pip-tools
	pip-compile --upgrade requirements/dev.in -o requirements/dev.txt
	pip-compile --upgrade requirements/prod.in -o requirements/prod.txt

ensure-pip-tools:
	pip install pip-tools

# Install dependencies
install-prod:
	pip install -r requirements/prod.txt
install-dev:
	pip install -r requirements/dev.txt

# Clean up build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Run linter
lint:
	flake8 .

# Run code format
format:
	black .
	isort .

# Build docker image
build:
	sudo docker build -t ndi_record:`git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_\-]/_/g'` -t ndi_record:latest .

# Run docker container
# run:
# 	sudo docker run -it --rm --gpus all --runtime=nvidia --network host --privileged \
# 	-v /var/run/dbus:/var/run/dbus \
# 	-v /run/avahi-daemon/socket:/run/avahi-daemon/socket \
# 	-v /srv/sftp/RECORDINGS:/app/output/ \
# 	-v ./:/app/ \
# 	ndi_record:`git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_\-]/_/g'`

run:
	sudo docker run -it --rm --gpus all --runtime=nvidia -p 8000:8000 --privileged \
	-v /var/run/dbus:/var/run/dbus \
	-v /run/avahi-daemon/socket:/run/avahi-daemon/socket \
	-v /srv/sftp/RECORDINGS:/app/output/ \
	-v ./:/app/ \
	ndi_record:`git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_\-]/_/g'`

run-xhost:
	sudo docker run -it --rm --gpus all --runtime=nvidia --network host --privileged \
	-v /var/run/dbus:/var/run/dbus \
	-v /run/avahi-daemon/socket:/run/avahi-daemon/socket \
	-v /srv/sftp/RECORDINGS:/app/output/ \
	-v ./:/app/ \
	--env="DISPLAY=$$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	ndi_record:`git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_\-]/_/g'`
