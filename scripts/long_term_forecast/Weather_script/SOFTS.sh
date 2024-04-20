model_name=SOFTS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1