#!/bin/bash
for i in $(find -name \*.gif); do
    convert "$i" "${i:0:-4}".png
done
