#!/bin/bash

# Initial values
x=0
y=2

# Loop until the command fails
while true; do
    # Execute the command
    verify-characteristic --rounds-from-to $x $y present80 trails/present/present_Wang08_table_7_r14.npz count-prob

    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "Execution failed with x=$x and y=$y"
        break
    fi

    # Increment x and y
    ((x++))
    ((y++))
done