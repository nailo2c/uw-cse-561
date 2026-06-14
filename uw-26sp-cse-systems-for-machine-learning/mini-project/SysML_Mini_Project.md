## **Building, characterizing, and optimizing an Agentic Application**

Students are tasked with architecting, building, and analyzing an agentic application. These systems move beyond simple chatbot interactions, requiring the agent to reason, use tools, and make autonomous decisions to achieve complex goals. Your project should implement a common agentic design pattern – such as reflection/self-correction, planning & decomposition, multi-agent collaboration, or dynamic tool orchestration – to solve a non-trivial problem. The objectives are: (1) To characterize agentic application performance at the application level — identifying the right metrics (e.g., end-to-end latency) for a given design pattern. (2) To analyze the systems-level implications of that design pattern by profiling inference engine performance (e.g., throughput) and identifying optimization opportunities that improve application-level metrics. 

### Project Stages

1. Project proposal (Due May 18): Students submit a write-up outlining  
   1. **The Application:** What is the application that will be prototyped? Which design pattern will be used?   
   2. **The Execution Plan:** How will the prototyping be performed? Here is some [guidance](https://gitlab.cs.washington.edu/zezhou/vllm-tpu-starter) (setup and simple client code) on running standard inference engines, such as vLLM, on Cloud TPUs.  
   3. **Hypothesis:** What do you expect the primary system bottleneck to be? What might be the optimization opportunities?  
2. Prototype development and profiling: Building and measuring a functional prototype of the agent.  
   1. Students implement the agent logic, tool integrations, and state management.  
   2. The prototype must be functional enough to complete an appropriately designed benchmark or trace.  
   3. Report measurements that characterize the performance of the agent  
   4. Perform a conceptual analysis of what are the optimization opportunities in a fully deployed system  
3. Project writeup and code (Due June 11): Submit a workshop-style paper (e.g., 6 pages, double-column writeup) describing the system design, performance results, and an analysis of the optimization opportunities. Also submit your code along with the writeup.

### Resources

Useful resources for learning more about ML agents.

* A free short [course](https://learn.deeplearning.ai/courses/agentic-ai/information) that covers the key design patterns (Reflection, Tool Use, Planning, and Multi-agent).  
* [Blog](https://medium.com/@maximilian.vogel/mastering-ai-agents-the-10-best-free-courses-tutorials-learning-tools-46bc380a19d1) post with various pointers to different types of agents.  
* Huggingface [course](https://huggingface.co/learn/agents-course/en/unit0/introduction) on AI agents.  
* Specific agents worth looking at:  
  * [Marketing agent](https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/orchestrator_workers.ipynb)   
  * [Financial agent](https://github.com/anthropics/claude-cookbooks/blob/main/multimodal/using_sub_agents.ipynb)  
  * [Deep researcher agent](https://github.com/anthropics/claude-cookbooks/tree/main/patterns/agents/prompts)  
  * [Customer service agent](https://github.com/openai/swarm/blob/main/examples/support_bot/README.md)   
  * [More customer service and handoffs](https://github.com/openai/openai-cookbook/blob/main/examples/Orchestrating_agents.ipynb)  
  * [Summarization for agentic conversations](https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/session_memory.ipynb)  
  * [Text to SQL agent](https://smolagents.org/docs/text-to-sql-example/)   
  * [Travel agent](https://huggingface.co/blog/smolagents)

