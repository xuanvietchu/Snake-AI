# Snake-AI
Implementation of Deep Q-Learning with Planning for Snake Game. After 1200 games, the snake achieves an average score of 15.56 and a best score of 40 for the 10x10 map.

![image](https://github.com/xuanvietchu/Snake-AI/assets/38886630/74625b4a-b4ba-4497-b9ff-9e996d22d37b)

This code is heavily based on the tutorial by freeCodeCamp at https://www.youtube.com/watch?v=L8ypSXwyBds&t=4187s.

This version of the code performs parallelized Q-value updates (much faster training time), is more streamlined for hyperparam search, and uses a more advanced dynamic reward system and Neural Network architecture. This architecture much outperforms other raw feature inputs such as CNN and specific cordinates.
