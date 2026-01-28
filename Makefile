# AlayaLite Makefile
# Usage: make help

.PHONY: help build build-debug build-release test test-cpp test-py lint clean install dev-install wheel

# Configuration
BUILD_DIR      := build
BUILD_TYPE     := Release
CMAKE_FLAGS    := -DBUILD_TESTING=ON
PYTEST_FLAGS   := -v
CTEST_FLAGS    := --output-on-failure
JOBS           := $(shell nproc 2>/dev/null || echo 4)
PYTHON_VERSION := 3.11

# Suppress "Entering/Leaving directory" messages from sub-makes
MAKEFLAGS     += --no-print-directory

# Colors
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(CYAN)AlayaLite Development Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make build          # Build release version"
	@echo "  make test           # Run all tests"

# ============================================================================
# Build
# ============================================================================

build: build-release ## Build project (release mode)

build-debug: ## Build project in debug mode
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) -j$(JOBS)

build-release: ## Build project in release mode
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) -j$(JOBS)

build-coverage: ## Build with coverage instrumentation
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) -DENABLE_COVERAGE=ON
	@cmake --build $(BUILD_DIR) -j$(JOBS)

configure: ## Only configure cmake (no build)
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)

# ============================================================================
# Test
# ============================================================================

test: test-cpp test-py ## Run all tests (C++ and Python)

test-cpp: build ## Run C++ tests with CTest
	@cd $(BUILD_DIR) && ctest $(CTEST_FLAGS) -j$(JOBS)

test-py: ## Run Python tests with pytest
	@uv run pytest $(PYTEST_FLAGS)

test-py-cov: ## Run Python tests with coverage
	@uv run pytest $(PYTEST_FLAGS) --cov=python/src --cov-report=html

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run all pre-commit checks
	@uvx pre-commit run -a

# ============================================================================
# Install & Package
# ============================================================================

install: ## Install package (production only)
	@uv sync --no-dev

dev-install: ## Install package with dev dependencies
	@uv sync

wheel: ## Build wheel package (use PYTHON_VERSION=3.x to specify)
	@uvx --python $(PYTHON_VERSION) --from build pyproject-build

# ============================================================================
# Clean
# ============================================================================

clean: ## Remove build artifacts
	@rm -rf $(BUILD_DIR)
	@rm -rf dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-all: clean ## Remove build artifacts and dependencies
	@rm -rf .venv
	@echo "Cleaned all."
