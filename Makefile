PYTHON=python

.PHONY: help deps pred train lint

help:
	@echo " make deps    Install python deps"
	@echo " make pred    Run daily prediction"
	@echo " make train   Run model training"
	@echo " make lint    Run a simple flake8 check if installed"

deps:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

pred:
	./run_prediction.sh

train:
	cd training && ./run_train.sh

lint:
	if command -v flake8 >/dev/null 2>&1; then \
		flake8 .; \
	else \
		echo "flake8 not installed; run 'make deps' or install flake8 manually"; \
	fi
