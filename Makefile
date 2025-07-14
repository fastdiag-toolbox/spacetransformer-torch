.PHONY: help install test build clean version

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Install package in development mode"
	@echo "  test       - Run tests"
	@echo "  build      - Build package"
	@echo "  clean      - Clean build artifacts"
	@echo "  version    - Show current version"

# Install package in development mode
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Build package
build:
	python -m build

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf spacetransformer/torch/_version.py
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +

# Show current version
version:
	@echo "Current version tag:"
	@git describe --tags --abbrev=0 2>/dev/null || echo "No tags found"
	@echo ""
	@echo "To create a new release:"
	@echo "  git tag v1.0.0"
	@echo "  git push origin v1.0.0" 