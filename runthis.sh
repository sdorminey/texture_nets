echo hi
sleep 10
 th train.lua -data /home/mabel/flock/dataset -style_image /home/mabel/flock/worker/Trainer_Tima/style/clouds.jpg -num_iterations 250 -learning_rate 0.040000 -out /home/mabel/flock/worker/Trainer_Tima/checkpoint/clouds_1.t7 -backend cudnn -batch_size 1 -model starling -style_weight 1
sleep 10
