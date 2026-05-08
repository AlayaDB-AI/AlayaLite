# AlayaLite Makefile
# Usage: make help

.PHONY: help build build-debug build-release build-san build-coverage \
        test test-cpp test-cpp-debug test-san test-py test-py-integration test-py-cov \
        lint format configure conan-install conan-install-debug \
        install dev-install wheel clean clean-release clean-debug clean-all

# Configuration
BUILD_RELEASE_DIR := build/Release
BUILD_DEBUG_DIR   := build/Debug
BUILD_SAN_DIR     := build/San
BUILD_TYPE        ?= Release
CMAKE_FLAGS       := -DBUILD_TESTING=ON
PYTEST_FLAGS      := -v
CTEST_FLAGS       := --output-on-failure -LE performance
JOBS              := $(shell nproc 2>/dev/null || echo 4)
PYTHON_VERSION    ?=

# Suppress "Entering/Leaving directory" messages from sub-makes
MAKEFLAGS += --no-print-directory

# Colors
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(CYAN)AlayaLite Development Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make build                  # Configure + build release"
	@echo "  make test                   # Run all tests (C++ release + Python)"
	@echo "  make build-san test-san     # Sanitizer build + test"
	@echo "  make configure BUILD_TYPE=Debug"

# ============================================================================
# Build
# ============================================================================

build: build-release ## Build project (release mode)

build-release: ## Configure + build in Release mode
	@cmake -B $(BUILD_RELEASE_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	@cmake --build $(BUILD_RELEASE_DIR) -j$(JOBS)

build-debug: ## Configure + build in Debug mode
	@cmake -B $(BUILD_DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DEBUG_DIR) -j$(JOBS)

build-san: ## Configure + build with ASan + UBSan (-O1 overrides -Ofast)
	@cmake -B $(BUILD_SAN_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) \
		-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1"
	@cmake --build $(BUILD_SAN_DIR) -j$(JOBS)

build-coverage: ## Configure + build with coverage instrumentation
	@cmake -B $(BUILD_DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) -DENABLE_COVERAGE=ON
	@cmake --build $(BUILD_DEBUG_DIR) -j$(JOBS)

configure: ## Configure only; override with BUILD_TYPE=Debug
	@cmake -B build/$(BUILD_TYPE) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)

conan-install: ## Re-run conan install for Release (after conanfile changes)
	@uv run python scripts/conan_build/conan_install.py --build-type Release

conan-install-debug: ## Re-run conan install for Debug
	@uv run python scripts/conan_build/conan_install.py --build-type Debug

# ============================================================================
# Test
# ============================================================================

test: test-cpp test-py ## Run all tests (C++ release + Python)

test-cpp: build-release ## Run C++ tests (release build)
	@ctest --test-dir $(BUILD_RELEASE_DIR) $(CTEST_FLAGS) -j$(JOBS)

test-cpp-debug: build-debug ## Run C++ tests (debug build)
	@ctest --test-dir $(BUILD_DEBUG_DIR) $(CTEST_FLAGS) -j$(JOBS)

test-san: build-san ## Run C++ tests under ASan + UBSan (single-threaded)
	@ASAN_OPTIONS=detect_leaks=1 ctest --test-dir $(BUILD_SAN_DIR) $(CTEST_FLAGS) -j1

test-py: ## Run Python unit/mock tests (fast, excludes integration)
	@uv run pytest $(PYTEST_FLAGS)

test-py-integration: ## Run Python integration tests (builds real indices; requires LASER build)
	@uv run pytest $(PYTEST_FLAGS) -m integration

test-py-cov: ## Run Python tests with HTML coverage report
	@uv run pytest $(PYTEST_FLAGS) --cov=python/src --cov-report=html

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run all pre-commit checks
	@uvx pre-commit run -a

format: ## Auto-format C++ sources with clang-format
	@find include tests tools python/src -name '*.h' -o -name '*.cpp' -o -name '*.cc' \
		| xargs clang-format -i --style=file

# ============================================================================
# Install & Package
# ============================================================================

install: ## Install package (production deps only)
	@uv sync --no-dev

dev-install: ## Install package with all dev dependencies
	@uv sync

wheel: ## Build wheel (PYTHON_VERSION=3.x to target a specific interpreter)
	@uv build $(if $(PYTHON_VERSION),--python $(PYTHON_VERSION))

# ============================================================================
# Clean
# ============================================================================

clean: ## Remove all build directories and caches
	@rm -rf build
	@rm -rf dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-release: ## Remove only Release build directory
	@rm -rf $(BUILD_RELEASE_DIR)

clean-debug: ## Remove only Debug/San build directories
	@rm -rf $(BUILD_DEBUG_DIR) $(BUILD_SAN_DIR)

clean-all: clean ## Remove build artifacts and virtualenv
	@rm -rf .venv
	@echo "Cleaned all."
