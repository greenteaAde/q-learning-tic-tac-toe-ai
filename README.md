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

* agent_RL : ��ȭ�н��� ���� �� ������Ʈ
* agent_Base : ����/�̱�� ���� �δ� �񱳿� ������Ʈ
* agent_Human : input�� �޾Ƽ� ���� ���� ������Ʈ
* policy(state) : state�� �޾Ƽ� e-greedy ��å�� ��ȯ�մϴ�.

## Environment

* Tictactoe ���� ȯ���Դϴ�.
* step(action) : action�� �޾Ƽ� �����ϰ� observation�� ��ȯ�մϴ�.
* render() : ���� ���¸� ȭ�鿡 ����մϴ�.
* init()/reset() : ȯ���� �ʱ�ȭ�մϴ�.

## Learning Algorithm

* Table�� �̿��� Q-learning(off-policy TD control)�� ����Ͽ����ϴ�.
* 500 Episode�� �ڰ��������� Training�մϴ�.
* 500 Episode���� agent_Base�� 100 Episode�� �׽�Ʈ�մϴ�.

* �н� ��

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

* agent_Base���� �������� �·��� 100%�� �����߽��ϴ�.
* ����� ����� ����(unbeatable)���� �н��� �Ǿ����ϴ�.

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning : An Introduction. 
The MIT Press Cambridge, Massachusetts London, England
