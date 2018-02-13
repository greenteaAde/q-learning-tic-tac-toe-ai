Tictactoe AI using Q-learning Method
=====================
## ����

#Training

```
$ python Train.py
```

#Play games

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
* �ڰ����� 350 Episode, ���̽����� 150 Episode�� ������ Training�մϴ�.
* 500 Episode���� agent_base�� 100 Episode�� �׽�Ʈ�մϴ�.
* 50000 Episode���� ��ս·� ����մϴ�.

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
* ����� ����� ����(unbeatable)���� �н���Ű�µ� �����߽��ϴ�.

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning : An Introduction. 
The MIT Press Cambridge, Massachusetts London, England
