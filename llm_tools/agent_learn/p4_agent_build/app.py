from planandsolveagent import PlanAndSolveAgent
from reactagent import HelloAgentsLLM, ReActAgent, ToolExecutor
from reflectionagent import ReflectionAgent

if __name__ == "__main__":
    pass
    # 初始化智能体
    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()

    # agent = ReActAgent(llm_client, tool_executor)
    # agent.run("今天吃什么好呢")

    # question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    # agent = PlanAndSolveAgent(llm_client)
    # agent.run(question)

    question = "编写一个Python函数，找出1到n之间所有的素数"
    agent = ReflectionAgent(llm_client)
    agent.run(question)
