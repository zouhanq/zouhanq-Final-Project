# Use bash for inline commands
SHELL := /bin/bash

# Name of the virtual environment directory
VENV_DIR := .venv

.PHONY: install run test clean

install:
	# Create a virtual environment and install dependencies
	python3 -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate; pip install --upgrade pip; pip install -r requirements.txt

run:
	# Run the main script inside the virtual environment
	. $(VENV_DIR)/bin/activate; python main.py

test:
	# Run tests inside the virtual environment
	. $(VENV_DIR)/bin/activate; pytest tests/

clean:
	# Clean up the virtual environment and any __pycache__ directories
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
