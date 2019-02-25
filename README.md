# deepq-collision-avoidance
deep q learning applied to collision avoidance

# todo

- add experience replay (last n frames)
- test blitting using matplotlib (it is way faster)
- gpu acceleration

# helping it learn

currently the problem is that the sampling of rewards causes the agent not to learn the general Q function (too many +1s, -1s are sparse)

- tweak reward r_n to be relative to distance (?)
- tweak exploration
- run it for longer (and on the GPU, let the memory get up to 1m+)
- make the boundaries another input feature (separate tensor), or encode the distance to each one (?)
- make the player's position (x, y) normalized from 0 to 1 as an input feature?
    - should still be able to learn that it's always in the cneter though
    
- ensure the distribution of +1 rewards to -1 rewards matches in sampling
    - OR, just tweak the reward function so that the distribution for both is similar
    - -1 if moving forward that way for N timesteps collides with an enemy and +1 if it doesn't
    - [HER?](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305_) (hindsight experience replay)
        - there is also [PER](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)
    - [convnetjs](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
    - make the -1 reward more likely