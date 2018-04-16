#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo Usage: $0 source_dir
    exit
fi

source_dir=$1
export source_dir

function convert_to_jpg
{
    gm convert "$source_dir/$1" -sampling-factor 4:2:0 -strip -resize 256x256^ -quality 82 -colorspace RGB "images/${1%%.*}.jpg" 2>&1
    if [ "$?" -ne 0 ]; then
        cp "$source_dir/$1" failed/
    fi
}

export -f convert_to_jpg

mkdir images
mkdir failed
parallel -a images.csv convert_to_jpg > resize_images.log
