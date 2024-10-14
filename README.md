# federated-learning-project

This project simulates training in a federated setting using the Flower framework. Data poisoining can be enabled to illustrate the effects of attacks on federated learning systems.

## Instructions

1. Download the instructor-provided FEMNIST dataset and named it `femnist_subset`
2. Create a new virtual environment and install the dependencies in `requirements.txt`
3. Start the regular unpoisoned simulation by running this command

```
flwr run
```

3. Then, try the poisoned simulation by overriding the `poison` option

```
flwr run --run-config "poison='true'"
```

## Group 3 Members

- Anh Nguyen
- Ananya Misra
- Chhabi Gautam
