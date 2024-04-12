#! /usr/bin/bash

mkdir build bin lib
cd build
rm -rf ./*
cmake ..
make -j16
cd ..