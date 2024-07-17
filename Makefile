upgrade-pip:
	python -m pip install --upgrade pip setuptools wheel pip-tools
compile-requirements:	
	python -m piptools compile requirements/dev_requirements.in --output-file requirements/dev_requirements.txt
install-requirements:
	python -m pip install -r requirements/dev_requirements.txt

install-all:
	python -m pip install --upgrade pip setuptools wheel
	python -m piptools compile requirements/dev_requirements.in --output-file requirements/dev_requirements.txt
	python -m pip install -r requirements/dev_requirements.txt
