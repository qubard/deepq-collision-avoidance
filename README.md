# deepq-collision-avoidance
deep q learning applied to collision avoidance

# todo

- add experience replay (last n frames)
- convert pygame to numpy blitting (should be faster)
- gpu acceleration
- tweak reward r_n to be relative to distance
- make the boundaries another input feature (separate tensor), or encode the distance to each one (?)
- make the player's position (x, y) normalized from 0 to 1 as an input feature?
    - should still be able to learn that it's always in the cneter though
    
- ensure the distribution of +1 rewards to -1 rewards matches in sampling
    - OR, just tweak the reward function so that the distribution for both is similar
    - -1 if moving forward that way for N timesteps collides with an enemy and +1 if it doesn't
    - [HER?](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305_) (hindsight experience replay)
    - [convnetjs](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)