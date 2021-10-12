#!/usr/bin/env bash

echo "____[Executing all unit test suits!]____"

testsuite_counter=0
#go through all dir under test
for dir in "test"/*; 
do
    if [ -d "$dir" ];
    then
    #save sub module name
    sub_module=$(echo $dir| cut -d'/' -f 2)
    #go through all paths in each sub dir
    for path in "$dir"/*
    do
        #filter out all valid files
        if [ -f "$path" ]  && [[ "$path" =~  "test_" ]]; 
        then
            #assemble test module name
            test=${path%.py}
            full_test_module="test.${sub_module}.${test##*/}"
            #execute test module
            python -m "$full_test_module"
            let testsuite_counter++
        fi
    done
    fi
done

echo ""
echo "_____[${testsuite_counter} test suits have been run!]____"
