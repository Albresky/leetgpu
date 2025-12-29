SHELL := /bin/bash
NVCC := nvcc
# Adjust architecture if needed. sm_89 is for RTX 4090.
# NVCC_FLAGS := -O3 -arch=sm_89 --std=c++17
NVCC_FLAGS := -O0 -arch=sm_89 --std=c++17

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Helper to find the directory for a given ID (e.g. 01 -> 01-vecadd)
FIND_DIR = $(shell find . -maxdepth 1 -type d -name "$(1)-*" | head -n 1)

.PHONY: all clean bench help

all: help

help:
	@echo "Usage:"
	@echo "  make <id>        Build and run the problem with the given ID (e.g., make 01)"
	@echo "  make bench <id>  Build and run with Nsight Systems profiling (e.g., make bench 01)"
	@echo "  make ptx <id>    Generate PTX assembly (e.g., make ptx 01)"
	@echo "  make sass <id>   Generate SASS assembly (e.g., make sass 01)"

clean:
	rm -f *_runner *.qdrep *.nsys-rep *.sqlite
	find . -name "*_runner" -delete
	find . -name "*.nsys-rep" -delete
	find . -name "*.sqlite" -delete
	find . -name "*.log" -delete

# Check if 'bench' is in the command line args
ifneq ($(filter bench,$(MAKECMDGOALS)),)
BENCH := 1
endif

# Check if 'ptx' is in the command line args
ifneq ($(filter ptx,$(MAKECMDGOALS)),)
PTX := 1
endif

# Check if 'sass' is in the command line args
ifneq ($(filter sass,$(MAKECMDGOALS)),)
SASS := 1
endif

# Check if 'clean' is in the command line args
ifneq ($(filter clean,$(MAKECMDGOALS)),)
CLEAN := 1
endif

# 'bench' target does nothing itself, just used to set the flag
bench:
	@if [ "$(MAKECMDGOALS)" = "bench" ]; then \
		echo "Error: Usage 'make bench <id>'"; \
		exit 1; \
	fi

ptx:
	@if [ "$(MAKECMDGOALS)" = "ptx" ]; then \
		echo "Error: Usage 'make ptx <id>'"; \
		exit 1; \
	fi

sass:
	@if [ "$(MAKECMDGOALS)" = "sass" ]; then \
		echo "Error: Usage 'make sass <id>'"; \
		exit 1; \
	fi

clean:
	@if [ "$(MAKECMDGOALS)" = "clean" ]; then \
		echo "Error: Usage 'make clean <id>'"; \
		exit 1; \
	fi

# Catch-all pattern rule for the ID
%:
	@# Find the directory matching the ID
	$(eval DIR := $(call FIND_DIR,$@))
	@if [ -z "$(DIR)" ]; then \
		echo "Error: No directory found for ID '$@'"; \
		exit 1; \
	fi
	@echo "Found directory: $(DIR)"
	
	@if [ "$(CLEAN)" = "1" ]; then \
		echo "Cleaning $(DIR)..."; \
		rm -f $(DIR)/*_runner $(DIR)/*.qdrep $(DIR)/*.nsys-rep $(DIR)/*.sqlite $(DIR)/*.log $(DIR)/*.ptx $(DIR)/*.sass $(DIR)/*.cubin; \
	elif [ "$(PTX)" = "1" ] || [ "$(SASS)" = "1" ]; then \
		if [ "$(PTX)" = "1" ]; then \
			echo "Generating PTX for $(DIR)..."; \
			$(NVCC) $(NVCC_FLAGS) -ptx $(DIR)/*.cu -o $(DIR)/$@.ptx; \
		fi; \
		if [ "$(SASS)" = "1" ]; then \
			echo "Generating SASS for $(DIR)..."; \
			$(NVCC) $(NVCC_FLAGS) -cubin $(DIR)/*.cu -o $(DIR)/$@.cubin; \
			cuobjdump -sass $(DIR)/$@.cubin > $(DIR)/$@.sass; \
			rm -f $(DIR)/$@.cubin; \
		fi; \
	else \
		echo "Compiling..."; \
		$(NVCC) $(NVCC_FLAGS) -o $(DIR)/$@_runner main.cu $(DIR)/*.cu; \
		if [ "$(BENCH)" = "1" ]; then \
			echo "Running benchmark (nsys)..."; \
			nsys profile --stats=true --force-overwrite=true -o $(DIR)/$@_report $(DIR)/$@_runner 2>&1 | tee $(DIR)/benchmark_$(TIMESTAMP).log; \
		else \
			echo "Running..."; \
			$(DIR)/$@_runner 2>&1 | tee $(DIR)/run_$(TIMESTAMP).log; \
		fi; \
		rm -f $(DIR)/$@_runner; \
	fi
