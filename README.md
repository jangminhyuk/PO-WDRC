Wasserstein Distributionally Robust Control of Partially Observable Linear Systems
====================================================

This repository includes the source code for implementing Wasserstein distributionally robust (WDR) and linear-quadratic-Gaussian (LQG) and Partially Observable Wasserstein Distributionally Robust (PO_WDR) controllers.

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- pickle5
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)**

## Usage

To run the experiments, call the main python script:
```
python main.py
```

The parameters can be changed by adding additional command-line arguments:
```
python main.py --dist normal --sim_type multiple --num_sim 1000 --num_samples 5 --horizon 50 --plot
```

The results for both controllers are saved in separate pickle files in `/results/<dist>/<noise_dist>/<sim_type>/`. If command-line argument `--plot` is invoked, the state, control, output trajectories, and cost histograms are plotted.


An example output with default settings:

```
Infimum penalty: 1.1683645425364375
Optimal penalty (lambda_star): 3.0019233984454297
---------------------
i:  0
cost (WDRC): 4.88131377528821 time (WDRC): 1.3135731220245361
cost (LQG): 13.35851384635249 time (LQG): 1.3135731220245361
i:  1
cost (WDRC): 4.815552342211396 time (WDRC): 1.447786569595337
cost (LQG): 5.899643756673218 time (LQG): 1.447786569595337
i:  2
cost (WDRC): 3.614633194940985 time (WDRC): 1.2856733798980713
cost (LQG): 5.209145042872055 time (LQG): 1.2856733798980713
=======
Optimal penalty (lambda_star): 167.8803891991265
---------------------
i:  0
cost (WDRC): 5.532708460900256 time (WDRC): 0.020344018936157227
cost (LQG): 13.236438720521598 time (LQG): 0.020344018936157227
i:  1
cost (WDRC): 5.890649205194933 time (WDRC): 0.018655776977539062
cost (LQG): 6.290175570638221 time (LQG): 0.018655776977539062
i:  2

...

i:  998
cost (WDRC): 4.617803231404117 time (WDRC): 1.1495120525360107
cost (LQG): 8.92887009759381 time (LQG): 1.1495120525360107
i:  999
cost (WDRC): 4.7021424455645136 time (WDRC): 1.1528325080871582
cost (LQG): 4.444349897961563 time (LQG): 1.1528325080871582

-------Summary-------
cost: 4.598837463514627 (0.5570802119654283) cost_lqr:5.3736445547841205 (1.3975270334306256)
time: 1.291071789264679 (0.09297734740459915) time_lqr: 0.02040037965774536 (0.005255226614429819)

=======
cost (WDRC): 4.389570592012172 time (WDRC): 0.017394304275512695
cost (LQG): 4.1788425472507615 time (LQG): 0.017394304275512695
i:  999
cost (WDRC): 5.512838777480511 time (WDRC): 0.018067359924316406
cost (LQG): 4.711580502602004 time (LQG): 0.018067359924316406

-------Summary-------
cost: 5.10159645316663 (0.9704279389511081) cost_lqr:5.645574012395052 (1.4601609716409711)
time: 0.017900517702102662 (0.0008626328210797351) time_lqr: 0.01762101650238037 (0.0007827119986816)
```

