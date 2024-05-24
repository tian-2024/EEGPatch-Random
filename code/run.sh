exp_time=$(date "+%y/%m/%d/%H%M")
log_dir="log_line/${exp_time}/"
mkdir -p "$log_dir"

sec=$(date "+%S")
model_name="resnet"
nohup python -u main.py --model ${model_name} --N 1 > ${log_dir}/${model_name}_N1_${sec}.log &
