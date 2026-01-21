让 Ai Agent 针对机器学习任务生成训练代码

Ai Agent工程做的事情就是写提示词让AI Agent更好地生成出训练代码

整个过程为:
1. 输出prompt让AI Agent生成计划(planning),训练代码(coding); 
2. 我们将上述planning, coding封装成一个节点node，表示AI Agent一个step得到的结果
3. 然后将代码提取出来交给解释器(Interpreter)执行，得到代码执行出来的结果/报错(我们统一称呼为反馈feedback);
4. 我们可以将node喂给AI Agent，让AI Agent依据训练代码和反馈进行评价， 此时node可以得到一个"标签"，Debug or Improve. (1)若反馈是报错则Debug (2)若反馈是结果则Improve，且有个AI Agent认为的得分。
5. 执行下一次step，AI Agent工程会选择出一个node进行下一步工作， 如果node标签为Debug则让AI Agent修复代码, 如果node标签为Improve则让AI Agent提升代码。此时又可以得到一个新node childnode,  此时childnode的父节点为node。
6. 重复上述过程我们可以得到一个AI Agent每个Step生成的node的tree。最后可以从叶子节点找到得分最大的结果。
7. 从tree上找node的规则为：(1)先满足人为设置的至少生成node个数。（此时不Debug还是Improve， 从Tree上理解就是多个头节点） (2)然后再优先找出需要DEBUG的节点 (3) 次优先找出目前得分最高的节点

帮助理解的[PPT](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw2.pdf)