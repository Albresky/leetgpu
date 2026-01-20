SHELL := /bin/bash
NVCC := nvcc

# arch: sm_89 for RTX 4090 (Ada Lovelance)
ARCH := 89
NVCC_FLAGS := -O0 -arch=sm_$(ARCH) --std=c++17
NVCC_DUMP_FLAGS := -O0 --std=c++17

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

FIND_DIR = $(shell find . -maxdepth 1 -type d -name "$(1)-*" | head -n 1)

.PHONY: all clean clean-all bench help ptx sass

all: help

help:
	@echo "Usage:"
	@echo "  make <id>        Build and run the problem with the given ID (e.g., make 01)"
	@echo "  make bench <id>  Build and run with Nsight Systems profiling (e.g., make bench 01)"
	@echo "  make ncu <id>	  Build and run with Nsight Compute profiling (e.g., make ncu 01)"
	@echo "  make ptx <id>    Generate PTX assembly (e.g., make ptx 01)"
	@echo "  make sass <id>   Generate SASS assembly (e.g., make sass 01)"

# Global clean (use: make clean-all)
clean-all:
	rm -f *_runner *.qdrep *.nsys-rep *.sqlite
	find . -name "*_runner" -delete
	find . -name "*.nsys-rep" -delete
	find . -name "*.ncu-rep" -delete
	find . -name "*.sqlite" -delete
	find . -name "*.log" -delete

# Check if 'bench' is in the command line args
ifneq ($(filter bench,$(MAKECMDGOALS)),)
BENCH := 1
endif

# Check if 'ncu' is in the command line args
ifneq ($(filter ncu,$(MAKECMDGOALS)),)
NCU := 1
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

ncu:
	@if [ "$(MAKECMDGOALS)" = "ncu" ]; then \
		echo "Error: Usage 'make ncu <id>'"; \
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
		echo "Error: Usage 'make clean <id>' or 'make clean-all' for global clean"; \
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
	$(eval SRC := $(shell ls $(DIR)/*.cu 2>/dev/null | grep -v _runner.cu | head -n 1))
	
	@if [ "$(CLEAN)" = "1" ]; then \
		echo "Cleaning $(DIR)..."; \
		rm -f $(DIR)/*_runner $(DIR)/*.qdrep $(DIR)/*.nsys-rep $(DIR)/*.ncu-rep $(DIR)/*.sqlite $(DIR)/*.log $(DIR)/*.ptx $(DIR)/*.sass $(DIR)/*.cubin; \
	elif [ "$(PTX)" = "1" ] || [ "$(SASS)" = "1" ]; then \
		if [ -z "$(SRC)" ]; then \
			echo "Error: No source file found in $(DIR)"; \
			exit 1; \
		fi; \
		echo "Source file: $(SRC)"; \
		if [ "$(PTX)" = "1" ]; then \
			echo "Generating PTX for $(SRC)..."; \
			$(NVCC) $(NVCC_DUMP_FLAGS) -arch=compute_$(ARCH) -ptx $(SRC) -o $(DIR)/$@.ptx; \
		fi; \
		if [ "$(SASS)" = "1" ]; then \
			echo "Generating SASS for $(SRC)..."; \
			$(NVCC) $(NVCC_DUMP_FLAGS) -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -cubin $(SRC) -o $(DIR)/$@.cubin; \
			cuobjdump -sass $(DIR)/$@.cubin > $(DIR)/$@.sass; \
			rm -f $(DIR)/$@.cubin; \
		fi; \
	else \
		echo "Compiling..."; \
		$(NVCC) $(NVCC_FLAGS) -o $(DIR)/$@_runner main.cu $(DIR)/*.cu; \
		if [ "$(BENCH)" = "1" ]; then \
			echo "Running benchmark (nsys)..."; \
			nsys profile --stats=true --force-overwrite=true -o $(DIR)/$@_report $(DIR)/$@_runner 2>&1 | tee $(DIR)/benchmark_$(TIMESTAMP).log; \
		elif [ "$(NCU)" = "1" ]; then \
			echo "Running benchmark (ncu)..."; \
			ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__inst_executed.sum,dram__throughput.avg_pct_of_peak_sustained_elapsed \
				$(DIR)/$@_runner 2>&1 | tee $(DIR)/ncu_$(TIMESTAMP).log; \
		else \
			echo "Running..."; \
			$(DIR)/$@_runner 2>&1 | tee $(DIR)/run_$(TIMESTAMP).log; \
		fi; \
	fi
