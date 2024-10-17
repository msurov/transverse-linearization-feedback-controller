#!/bin/bash
rm -rf build/*
rm -rf .venv/lib/python3.11/site-packages/car_trailers
rm -rf .venv/lib/python3.11/site-packages/car_trailers_traj_planner
rm -rf .venv/lib/python3.11/site-packages/common
rm -rf .venv/lib/python3.11/site-packages/demo
rm -rf .venv/lib/python3.11/site-packages/tests
rm -rf .venv/lib/python3.11/site-packages/car_trailers_demo
python -m pip install --no-dependencies .
