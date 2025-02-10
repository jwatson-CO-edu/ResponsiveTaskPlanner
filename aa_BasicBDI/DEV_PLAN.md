# Concepts
* Probabilistic Backchaining
    - Can I relate this to my makespan calculation?
* How are beliefs updated?
* How are the probabilities of outcomes updated?
* Priority Queue of Intents (Actions)
    - Priority: (Value of goal(s) depending on action) * (Joint probability that the best outcome is reached)
    - Pop from front of queue and execute
    - Does every action need an evaluator?
        * Only evaluate if we are insecure about success?
* How to handle periodic updates?

# Brainstorming
* Evaluate graphs with LLM agent?