import sys, subprocess

if __name__ == '__main__':
    
    input_json_file = sys.argv[1]
    output_pred_file = sys.argv[2]

    subprocess.call("python -u submission68_1.py --embd-matrix embd_weight --model-dir trained_model --input-file {} --output-file {} --batch-size 32 --embd-dp 0.2 --dropout 0.1 --gcn-dropout 0.1 --seed 999 --rnn-size 100 --num-hop 5 --lr 0.001 \
        --cand-edge 1 --all-ment-edge 1 --cand2ment-edge 1 --lr-reduction-factor 0.2 --word-dropout 0.0 --cm-fusion 1 --adapt-fusion 0 --scheduler cosine --t-initial 5".format(input_json_file, output_pred_file), shell=True)



