# Makefile for wildfire prediction project

init:
	conda env create -f environment.yml

update:
	conda env update -f environment.yml

repro:
	python -m dvc repro

train:
	make repro
	python pipelines/train_model.py

clean:
	dvc destroy
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf artifacts/figures/*
	rm -rf artifacts/models/*
	rm -rf artifacts/metrics/*

pull:
	git pull
	python -m dvc pull

push:
	git add .
	git commit -m "${MSG}" || true
	git push
	dvc commit
	dvc push
	git add dvc.lock .dvc/
	git commit -m "Update DVC tracking: ${MSG}" || true
	git push

.PHONY: init repro train clean push pull