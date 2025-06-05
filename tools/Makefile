PYTHON:= python3.9
CUDA_VERSION:= 12.1
PYTORCH_VERSION:= 2.3.1
DOT:= .
VENV_BASE:= venv
.PHONY: all clean show_variables

all: show_variables virtualenv.done pytorch.done parallel_wavegan.done

show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)
	@echo VENV_BASE=$(VENV_BASE)

virtualenv.done: show_variables
	test -d venv || $(PYTHON) -m venv --system-site-packages $(VENV_BASE)
	. $(VENV_BASE)/bin/activate; cd ../; pip install -U pip setuptools wheel
	# install numpy here since python3.6 is not supported in > 1.20
	. $(VENV_BASE)/bin/activate; cd ../; pip install numpy
	touch virtualenv.done

pytorch.done: virtualenv.done
ifeq ($(CUDA_VERSION),)
	. $(VENV_BASE)/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cpu/stable.html
else
	. $(VENV_BASE)/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
endif
	touch pytorch.done

parallel_wavegan.done: virtualenv.done pytorch.done
	. $(VENV_BASE)/bin/activate; cd ../; pip install -e .
	. $(VENV_BASE)/bin/activate; cd ../; pip install -e .[test]
	touch parallel_wavegan.done

apex.done: virtualenv.done pytorch.done
	git clone https://github.com/NVIDIA/apex.git
	. $(VENV_BASE)/bin/activate; cd apex; \
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	touch apex.done

clean:
	rm -r $(VENV_BASE) apex *.done
	find -iname "*.pyc" -delete
