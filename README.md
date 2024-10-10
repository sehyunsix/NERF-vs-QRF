# NERF-vs-QRF

# Task

To  test model reconsturction 2d image by postion input

# HOW TO

```bash
cd NERF-vs-QRF

#use python 3.12.6 version
pip install -r requreiment.txt
cd 2d-image

#revise config.py for set parameter
python3 main.py
```


# Result
## Test 1
|Original|Mlp|Postion|Quantum|
|---|---|---|---|
|![2d](2d-image/test1.jpg)|![2d](2d-image/results/predicted_test1.jpg_mlp.png)|![2d](2d-image/results/predicted_test1.jpg_position.png)|![2d](2d-image/results/predicted_test1.jpg_quantum.png)|


## Test 2
|Original|Mlp|Postion|Quantum|
|---|---|---|---|
|![2d](2d-image/test2.jpg)|![2d](2d-image/results/predicted_test2.jpg_mlp.png)|![2d](2d-image/results/predicted_test2.jpg_position.png)|![2d](2d-image/results/predicted_test2.jpg_quantum.png)|



## Test 3
|Original|Mlp|Postion|Quantum|
|---|---|---|---|
|![2d](2d-image/test3.jpg)|![2d](2d-image/results/predicted_test3.jpg_mlp.png)|![2d](2d-image/results/predicted_test3.jpg_position.png)|![2d](2d-image/results/predicted_test3.jpg_quantum.png)|

## Trainig Loss

|Mlp|Postion|Quantum|
|---|----|---|
|![2d](2d-image/loss_result/loss_graph_mlp.png)|![2d](2d-image/loss_result/loss_graph_position.png)|![2d](2d-image/loss_result/loss_graph_position.png)|


