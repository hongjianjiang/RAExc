# Makefile for Transformer Language Recognition Project

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  clean       - Remove all generated files (.pt, .pth, .log)"
	@echo "  clean-models - Remove only model files (.pt, .pth)"
	@echo "  clean-logs  - Remove only log files (.log)"
	@echo "  clean-data  - Remove dataset directories"
	@echo "  clean-all   - Remove all generated files and datasets"
	@echo "  train-l1    - Train L1 language model"
	@echo "  train-l2    - Train L2 language model"
	@echo "  train-l3    - Train L3 language model"
	@echo "  train-l4    - Train L4 language model"
	@echo "  train-l5    - Train L5 language model"
	@echo "  train-s1    - Train S1 language model"
	@echo "  test        - Run tests on generated data"

# Clean targets
.PHONY: clean
clean: clean-models clean-logs
	@echo "Cleaned all generated files"

.PHONY: clean-models
clean-models:
	@echo "Removing model files..."
	@find . -name "*.pt" -type f -delete
	@find . -name "*.pth" -type f -delete
	@echo "Model files removed"

.PHONY: clean-logs
clean-logs:
	@echo "Removing log files..."
	@find . -name "*.log" -type f -delete
	@echo "Log files removed"

.PHONY: clean-data
clean-data:
	@echo "Removing dataset directories..."
	@rm -rf dataset/
	@echo "Dataset directories removed"

.PHONY: clean-all
clean-all: clean-models clean-logs clean-data
	@echo "All generated files and datasets removed"

# Training targets
.PHONY: train-l1
train-l1:
	python main.py --language L1 --epochs 100

.PHONY: train-l2
train-l2:
	python main.py --language L2 --epochs 100

.PHONY: train-l3
train-l3:
	python main.py --language L3 --epochs 100

.PHONY: train-l4
train-l4:
	python main.py --language L4 --epochs 100

.PHONY: train-l5
train-l5:
	python main.py --language L5 --epochs 100

.PHONY: train-s1
train-s1:
	python main.py --language S1 --epochs 100

# Test target
.PHONY: test
test:
	python gen_benchmark.py

# Quick training with fewer epochs for testing
.PHONY: quick-train-l4
quick-train-l4:
	python main.py --language L4 --epochs 10 --early_stop_patience 5

.PHONY: quick-train-l5
quick-train-l5:
	python main.py --language L5 --epochs 10 --early_stop_patience 5

# Show file counts
.PHONY: status
status:
	@echo "Current file counts:"
	@echo "  .pt files:  $$(find . -name "*.pt" -type f | wc -l)"
	@echo "  .pth files: $$(find . -name "*.pth" -type f | wc -l)"
	@echo "  .log files: $$(find . -name "*.log" -type f | wc -l)"
	@echo "  Dataset dirs: $$(find . -name "dataset" -type d | wc -l)"
