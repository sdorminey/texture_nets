th test.lua -input /home/mabel/style/ostagram/content2.jpg -output /home/mabel/style/ostagram/$1_$2-n1.jpg -model_t7 /home/mabel/flock/worker/client/checkpoint/$1_$2.t7 -fader1 -0.9
th test.lua -input /home/mabel/style/ostagram/content2.jpg -output /home/mabel/style/ostagram/$1_$2-0.jpg -model_t7 /home/mabel/flock/worker/client/checkpoint/$1_$2.t7 -fader1 0
th test.lua -input /home/mabel/style/ostagram/content2.jpg -output /home/mabel/style/ostagram/$1_$2-1.jpg -model_t7 /home/mabel/flock/worker/client/checkpoint/$1_$2.t7 -fader1 1
display /home/mabel/style/ostagram/$1_$2-n1.jpg &
display /home/mabel/style/ostagram/$1_$2-0.jpg &
display /home/mabel/style/ostagram/$1_$2-1.jpg &
