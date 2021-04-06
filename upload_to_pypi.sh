#!/usr/bin/env bash

pytest -v test/
python setup.py sdist bdist_wheel
twine check dist/* 

read -p "Are you sure [y]? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    twine upload dist/*
fi

# pip install autograd-minimize -U 
# pip uninstall autograd-minimize 
# python setup.py develop