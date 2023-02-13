#!/bin/sh

echo "Do you want to train the network or predict a random image?"
TASK=$(gum choose "train" "predict")

mix $TASK
