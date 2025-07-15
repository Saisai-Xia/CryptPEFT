#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH
bandwidths=("100mbit" "500mbit" "1gbit" "5gbit")
delays=("4ms")
for bw in "${bandwidths[@]}"; do
    for delay in "${delays[@]}"; do
        sudo tc qdisc add dev lo root netem rate $bw delay $delay

        bash AE/ablation_LinAtten_PI.sh 0 $bw &

        pid0=$!

        bash AE/ablation_LinAtten_PI.sh 1 $bw &

        pid1=$!

        wait $pid0

        wait $pid1
        sudo tc qdisc del dev lo root
        echo "============================"
    done
done
python3 utils/plot.py