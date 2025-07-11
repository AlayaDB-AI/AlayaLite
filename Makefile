# Makefile for managing Python projects with uv
# This Makefile provides common tasks such as installing dependencies,
# running tests, linting, formatting, and cleaning.
# It leverages uv for faster dependency management.

# -----------------------------------------------------------------------------
# Project Configuration
# -----------------------------------------------------------------------------
PYTHON_VERSION ?= 3.11
PROJECT_NAME   := alayalite        # The name of your project
PYTHON         := .venv/bin/python  # Path to the Python interpreter in the virtual environment

# -----------------------------------------------------------------------------
# Colors for Terminal Output
# -----------------------------------------------------------------------------

# Use terminal colors for better readability (optional)
RED    := $(shell tput -Txterm setaf 1)  # Red    color
GREEN  := $(shell tput -Txterm setaf 2)  # Green  color
YELLOW := $(shell tput -Txterm setaf 3)  # Yellow color
BLUE   := $(shell tput -Txterm setaf 4)  # Blue   color
RESET  := $(shell tput -Txterm sgr0)     # Reset  color

# -----------------------------------------------------------------------------
# Target Definitions
# -----------------------------------------------------------------------------

# .PHONY: Declare targets that are not actual files
# This prevents conflicts with files that have the same name as the targets.
.PHONY: help uv-venv sync install test lint format clean

# Default target: 'help'
# When you run 'make' without any arguments, it will execute the 'help' target.
default: help

# Help target: Display available targets and their descriptions
help:
	@echo "$(BLUE)Makefile Help$(RESET)"
	@echo "-----------------------------------------------------------------------------"
	@echo "Available targets:"
	@echo "  sync   - Sync dependencies using uv."
	@echo "  install   - Install the project in editable mode."
	@echo "  build     - Build the project using uv."
	@echo "  test      - Run tests with pytest."
	@echo "  lint      - Run linters with ruff."
	@echo "  format    - Format code with ruff."
	@echo "  clean     - Clean the project (remove virtual environment, caches, etc.)."
	@echo "  help      - Show this help message."
	@echo "-----------------------------------------------------------------------------"
	@echo "$(YELLOW)Example: make test$(RESET)"

# sync target: Sync dependencies using uv
# This target creates the virtual environment if it doesn't exist and then syncs dependencies using uv.
sync:
	@echo "$(GREEN)Syncing dependencies with uv...$(RESET)"
	uv sync -p $(PYTHON_VERSION) --no-install-project --dev
	@echo "$(GREEN)Dependencies synced!$(RESET)"

# install target: Install the project in editable mode
# This allows you to make changes to the code without having to reinstall the project.
install: clean
	@echo "$(GREEN)Installing project in editable mode...$(RESET)"
	uv sync -p $(PYTHON_VERSION) --dev --reinstall -v
	@echo "$(GREEN)Project installed in editable mode!$(RESET)"

build: clean
	@echo "$(GREEN)Building project...$(RESET)"
	ALAYA_ROOT=$(shell dirname "$$(pwd)") uv build -p $(PYTHON_VERSION)
	@echo "$(GREEN)Project built!$(RESET)"

# test target: Run tests with pytest
# This target runs the tests using pytest.
test:
	@echo "$(GREEN)Running tests with pytest...$(RESET)"
	$(PYTHON) -m pytest
	@echo "$(GREEN)Tests finished!$(RESET)"

# lint target: Run linters with ruff
# This target checks the code for style issues using ruff.
lint: sync
	@echo "$(GREEN)Running linters with ruff...$(RESET)"
	$(PYTHON) -m ruff check .  # Run ruff
	@echo "$(GREEN)Linters finished!$(RESET)"

# format target: Format code with ruff
# This target automatically formats the code using ruff.
format: sync
	@echo "$(GREEN)Formatting code with ruff...$(RESET)"
	$(PYTHON) -m ruff format .
	@echo "$(GREEN)Code formatted!$(RESET)"

# clean target: Clean the project
# This target removes the virtual environment, cache files, and other temporary files.
clean:
	@echo "$(RED)Cleaning project...$(RESET)"
	@rm -rf CMakeUserPresets.json build .venv .pytest_cache .ruff_cache
	@rm -rf python/.pytest_cache python/.ruff_cache python/tests/__pycache__
	@echo "$(RED)Project cleaned!$(RESET)"

# -----------------------------------------------------------------------------
# Prevent command name conflicts
# -----------------------------------------------------------------------------
.PHONY: $(MAKECMDGOALS)
