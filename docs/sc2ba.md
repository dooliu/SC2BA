## Table of Contents

- [StarCraft II](#starcraft-ii)
    - [Micromanagement](#micromanagement)
- [SMAC](#smac)
    - [Scenarios](#scenarios)
    - [State and Observations](#state-and-observations)
    - [Action Space](#action-space)
    - [Rewards](#rewards)
    - [Environment Settings](#environment-settings)

## StarCraft II

SMAC is based on the popular real-time strategy (RTS) game [StarCraft II](http://us.battle.net/sc2/en/game/guide/whats-sc2) written by [Blizzard](http://blizzard.com/).
In a regular full game of StarCraft II, one or more humans compete against each other or against a built-in game AI to gather resources, construct buildings, and build armies of units to defeat their opponents.

Akin to most RTSs, StarCraft has two main gameplay components: macromanagement and micromanagement. 
- _Macromanagement_ (macro) refers to high-level strategic considerations, such as economy and resource management. 
- _Micromanagement_ (micro) refers to fine-grained control of individual units.

### Micromanagement

StarCraft has been used as a research platform for AI, and more recently, RL. Typically, the game is framed as a competitive problem: an agent takes the role of a human player, making macromanagement decisions and performing micromanagement as a puppeteer that issues orders to individual units from a centralised controller.

In order to build a rich multi-agent testbed, we instead focus solely on micromanagement.
Micro is a vital aspect of StarCraft gameplay with a high skill ceiling, and is practiced in isolation by amateur and professional players.
For SMAC, we leverage the natural multi-agent structure of micromanagement by proposing a modified version of the problem designed specifically for decentralised control.
In particular, we require that each unit be controlled by an independent agent that conditions only on local observations restricted to a limited field of view centred on that unit.
Groups of these agents must be trained to solve challenging combat scenarios, battling an opposing army under the centralised control of the game's built-in scripted AI.

Proper micro of units during battles will maximise the damage dealt to enemy units while minimising damage received, and requires a range of skills.
For example, one important technique is _focus fire_, i.e., ordering units to jointly attack and kill enemy units one after another. When focusing fire, it is important to avoid _overkill_: inflicting more damage to units than is necessary to kill them.

Other common micromanagement techniques include: assembling units into formations based on their armour types, making enemy units give chase while maintaining enough distance so that little or no damage is incurred (_kiting_), coordinating the positioning of units to attack from different directions or taking advantage of the terrain to defeat the enemy. 

Learning these rich cooperative behaviours under partial observability is challenging task, which can be used to evaluate the effectiveness of multi-agent reinforcement learning (MARL) algorithms.

## SC2BA

SC2BA uses the [StarCraft II Learning Environment](https://github.com/deepmind/pysc2) to introduce a competitive MARL environment which can benchmarking algorithms in adversary mode.

SC2BA provides an algorithm-vs-algorithm adversary evaluation platform by offering a powerful and scalable battle environment, along with two adversary modes, various combat scenarios, and other designs.

### Adversary Modes

To better verify the diversified and adversarial ability of MARL algorithms, we design two adversary modes: dual-algorithm paired adversary mode and multi-algorithm mixed adversary mode.

\textbf{Dual-algorithm paired adversary mode}. It introduces algorithm-vs-algorithm combat to enhance dynamic adversarial interactions, thus encouraging multi-agent to explore more robust winning strategies. In each scenario, agents are divided into two teams, each of which is controlled by a MARL algorithm. The two teams can continuously learn action policies and adapt to dynamic changes of the opponent. This requires MARL algorithms to possess robust adaptive and learning capabilities to gain an upper hand against ever-evolving opponents. Thus, this mode can evaluate the adversarial capability between two MARL algorithms.


\textbf{Multi-algorithm mixed adversary mode}. This mode is used to further evaluate the ability of a multi-agent algorithm when confronted with more diverse opponent actions. To implement this, we integrate multiple well-trained MARL models into SC2BA to alternately control the enemy troops. The opponent model is randomly selected from specified MARL algorithms for each episode, making the opponent's actions completely transparent and unpredictable. Thus, the agent algorithm should adapt to different opponents, which boosts them to master a wider range of tactical approaches. 

### Scenarios

In a regular full game of SC2, the players of two camps compete for resources, construct buildings, train armies, and eliminate all enemy buildings to achieve ultimate victory. Differ from the full game of SC2, we design a flexible team-adversarial game with the following specifications: each player controls a single combat unit instead of managing a complex macro-level strategy with an entire army; each team consists of multiple units that collaborate to fight against the opposing team to achieve final victory.

To evaluate multi-agent algorithms on scenario changes, we provide a series of meticulously designed combat scenarios, incorporating symmetric and asymmetric setups. The currently defined scenarios are listed in Table~\ref{SC2BA_scenarios}. They can be used to evaluate the ability of MARL algorithms when suffering diverse behaviors, rich dynamics, unequal troops and difficult collaboration. 

\textbf{Symmetric scenarios}. In symmetric scenarios, our goal is to create a balanced competitive platform. The composition and quantity of units for both teams are equal, ensuring a relatively balanced starting point. Symmetric scenarios are devoted to revealing MARL algorithms' inherent ability in learning coordination strategy and execution skills. 

\textbf{Asymmetric scenarios}. Asymmetric scenarios are more challenging due to non-equilibrium forces/layouts. For example, the enemy team possesses a quantity advantage in terms of troop strength, thereby intensifying the challenge of achieving victory. This setup necessitates MARL algorithms to demonstrate superior-level cooperation and remarkable operation skills to overcome asymmetric challenges.

The complete list of challenges is presented bellow. 

| Name | Ally Units | Enemy Units |
| :---: | :---: | :---:|
| 3m | 3 Marines | 3 Marines |
| 8m | 8 Marines | 8 Marines |
| 25m | 25 Marines | 25 Marines |
| MMM |  1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 2 Marauders & 7 Marines |
| 2s3z |  2 Stalkers & 3 Zealots |  2 Stalkers & 3 Zealots |
| 3s5z |  3 Stalkers &  5 Zealots |  3 Stalkers &  5 Zealots |
| 1c3s5z | 1 Colossi & 3 Stalkers & 5 Zealots | 1 Colossi & 3 Stalkers & 5 Zealots |
| 5m_vs_6m | 5 Marines | 6 Marines |
| 10m_vs_11m | 10 Marines | 11 Marines |
| MMM2 |  1 Medivac, 2 Marauders & 7 Marines |  1 Medivac, 3 Marauders & 8 Marines |

### State and Observations

In each time step of battle episode, agents can obtain observation information within their perception field from the environment. As used in SMAC, the sight range of all agents is also set to 9. It means that agents are partially observable in this setting. According to the sight range, we have the observation attributes of each agent as follows:

```python
observation_data = {
    'movements': [north, south, east, west],
    'enemies': [[enemy_id, distance, relative_x, relative_y, health, shield, unit_type], ...],
    'allies': [[distance, relative_x, relative_y, health, shield, unit_type], ...],
    'personal': [health, shield, unit_type]
}
```

The adversary states record the information of all agents from both teams at one-time slice, including the position of all units relative to the central point and other relevant features of agents. Please note that the adversary states are only available during the training phase. The details of the adversary state are outlined below.

```python
state_data = {
    'enemies': [[health, weapon_cd, relative_x, relative_y, shield, unit_type], ...],
    'allies': [[health, relative_x, relative_y, shield, unit_type], ...],
}
```

### Action Space

To simplify the vast and complex instruction set in SC2, we utilize a discrete set of actions that are automatically converted into executable instructions within the game engine. The action space is divided into three categories: move, attack, stop/no-op. The move action further contains four directions: north, south, east and west. The living agents can perform attack actions (when hostile agents are viewed and within attack range), or moving operations, or even stop operation (when choosing to maintain their current state). Dead agents can only perform no-op action.

In the simulated environment, the attack ranges for all units are set to a fixed value of 6, same to SMAC. If an agent executes the attack action out of actual attack range, it can not directly open fire. Instead, it triggers a built-in attack-move macro-actions on the target unit. When the enemies reach the attack radius, The agent will automatically go close to the specified enemy and start firing.

The battle units possess a diverse range of attributes and exhibit mutual counters. First, units possess different armor types: light and armored(heavy armor). Second, the arsenal encompasses a variety of weapons, with certain weapons bearing special effects. For instance, the marauder's punisher grenades inflict double damage against units with heavy armor, while the colossus is equipped with dual thermal lances that inflict area damage and extra damage against units with light armor.

### Rewards

To promote adversarial dynamics among algorithms, we redefine the reward function within SMAC. The reward function comprises two primary elements: immediate rewards and cumulative rewards. Firstly, for immediate rewards, we adhere to the design principles established in SMAC, where agents receive response rewards based on the current battle situation in each time step. These rewards consist of damaging/killing enemies as well as penalties for self-inflicted harm and death. Furthermore, with regard to the episodic rewards, we introduce penalties for failures and draws. For example, if neither of teams manages to eliminate all enemies within the pre-specified time limit, both teams receive a penalty reward. This setting carries meaning as it inspires an active participation mindset in both teams, encouraging them to passionately pursue victory rather than avoiding conflict out of fear of failure.

### Environment Settings

SC2BA makes use of the [StarCraft II Learning Environment](https://arxiv.org/abs/1708.04782) (SC2LE) to communicate with the StarCraft II engine. SC2LE provides full control of the game by allowing to send commands and receive observations from the game. However, SMAC is conceptually different from the RL environment of SC2LE. The goal of SC2LE is to learn to play the full game of StarCraft II. This is a competitive task where a centralised RL agent receives RGB pixels as input and performs both macro and micro with the player-level control similar to human players. SMAC, on the other hand, represents a set of cooperative multi-agent micro challenges where each learning agent controls a single military unit.

SC2BA uses the _raw API_ of SC2LE. Raw API observations do not have any graphical component and include information about the units on the map such as health, location coordinates, etc. The raw API also allows sending action commands to individual units using their unit IDs. This setting differs from how humans play the actual game, but is convenient for designing decentralised multi-agent learning tasks.

Since our micro-scenarios are shorter than actual StarCraft II games, restarting the game after each episode presents a computational bottleneck. To streamline the training/testing process, we automatically control the game stages through some code commands. This allows us to easily generate battle episodes with diverse unit compositions and environmental conditions based on specific configuration information. Furthermore, it automatically restarts a new episode upon the current one is terminated.

We can assign distinct multi-agent algorithm models to control the agents on both teams using commands. The starting point of algorithm models is set to random sampling or individual well-trained models (e.g., trained with built-in AI bots). During the training stage, we can set both teams to be controlled by algorithms capable of continuous evolution for injecting dynamic variations. Additionally, we can configure multiple well-trained models to alternately control the enemy troops, which could enrich the behavioral diversity to some extent.

