LANGS=(L1 L2 L3 L4 L5 L6 L7 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11)
SCRIPTS=("random_testing_last_letter.py" "random_testing_l1.py" "random_testing_edit.py" "random_testing_hamming.py")

for lang in "${LANGS[@]}"; do
  for script in "${SCRIPTS[@]}"; do
    # echo "Running $script for language $lang"
    python3 $script --language $lang --delta 1 --num_trials 1 --model transformer  
  done
done

 