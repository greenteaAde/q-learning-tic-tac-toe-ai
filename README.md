Tictactoe AI using Q-learning Method
=====================

## Run

### Training

```
$ python Train.py
```

### Play games

```
$ python Play.py
```

## Agents

* agent_RL : 강화학습을 진행 할 에이전트
* agent_Base : 랜덤/이기는 수를 두는 비교용 에이전트
* agent_Human : input을 받아서 수를 놓는 에이전트
* policy(state) : state를 받아서 e-greedy 정책을 반환합니다.

## Environment

* Tictactoe 게임 환경입니다.
* step(action) : action을 받아서 실행하고 observation을 반환합니다.
* render() : 현재 상태를 화면에 출력합니다.
* init()/reset() : 환경을 초기화합니다.

## Learning Algorithm

* Table을 이용한 Q-learning(off-policy TD control)을 사용하였습니다.
* 500 Episode의 자가대전으로 Training합니다.
* 500 Episode마다 agent_Base와 100 Episode씩 테스트합니다.

* 학습 식

```
Q(S,A) = Q(S,A) + learning_rate * [R + discount_factor * Max(Q(S',a)) - Q(S,A)]
```

* Hyperparameter

```
1. learning rate : 0.4
2. discount_factor : 0.9
3. epsilon (egreedy method) : 0.08
```

## Conclusion

* agent_Base와의 대전에서 승률이 100%에 도달했습니다.
* 사람과 비슷한 수준(unbeatable)까지 학습이 되었습니다.

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning : An Introduction. 
The MIT Press Cambridge, Massachusetts London, England
