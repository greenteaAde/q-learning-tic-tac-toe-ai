RL-based Tictactoe AI
=====================
## Agents

* agent_RL : 강화학습을 진행할 에이전트
* agent_Base : 랜덤/이기는 수를 두는 비교용 에이전트
* agent_Human : input을 받아서, 수를 놓는 에이전트

## Environment

* Tictactoe 게임 환경입니다.
* step(action) : action을 받아서 실행하고 observation을 반환합니다.
* render() : 현재 상태를 화면에 출력합니다.
* init()/reset() : 환경을 초기화합니다.

## Learning Algorithm

* Table을 이용한 Temporal-difference learning method를 사용하였습니다.
* 자가대전 350 Episode, 베이스대전 150 Episode로 Training합니다.
* 500 Episode마다 베이스와 100 Episode를 테스트합니다.
* 50000 Episode마다 평균승률 출력합니다.

* 학습 식

```
V(s) := V(s) + learning_rate * (V(s') - V(s))
```

* Hyperparameter

```
1. learning rate : 0.4
2. epsilon : 0.08
```

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning:An Introduction. 
The MIT Press Cambridge, Massachusetts London, England
