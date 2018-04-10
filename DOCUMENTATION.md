# Prioritised experience replay

I've already built a self driving taxi using a concept in machine reinforcement learning called uniform experience replay to aid in the process deep Q-learning. 

After coming across the paper below however, I've seen the amazing performance gains that this method of a more stratified sampling of replay memory can offer.

Check out the origional  [Google Deepmind 2016 PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/pdf/1511.05952v3.pdf) if you want a deeper understanding.

## So your taxi drives itself?
You might've gathered by now that this taxi is self driving but what does this mean? 

If you're not familiar with what machine learning (ML) is, I recommend you go and check out [my blog post](https://medium.com/@ross.nkama/an-intuitive-understanding-of-machine-learning-6814add2b2a9) to get a high level understanding. 

So I've been getting my machine intelligence's Neural Network to use a technique called reinforcement learning to incrementally update it's structure based off of it's ability to find correlations between the state that it's currently experiencing and (from experience) that state's ability to bring about particular rewards and consuquences at particular times.

These experiences I'm talking about are transitions in time where:  
$\hspace{1cm} transition = [s, a, s\prime, r]$  

Which is to say that a transition is where an action $a$, taken at a state $s$, leads to the agent recieving a reward $r$ (positive or negative) at the next state $s\prime$.