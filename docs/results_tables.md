### Plain protocol (3-seed ensembles)

| Grid | GateXer nMAE | Strongest baseline | Its nMAE | Gain % | 95% CI |
|---|---|---|---|---|---|
| ERCOT | 0.1497 | iTransformer | 0.1611 | 7.1 | [5.0, 9.2] |
| CAISO | 0.1451 | iTransformer | 0.1542 | 5.9 | [3.6, 8.0] |
| SWPP | 0.1548 | iTransformer | 0.1689 | 8.4 | [6.3, 10.4] |
| MISO | 0.1650 | TimeXer | 0.1752 | 5.8 | [4.0, 7.6] |
| Germany | 0.1795 | iTransformer | 0.1884 | 4.8 | [2.6, 6.9] |
| Spain | 0.1440 | iTransformer | 0.1496 | 3.8 | [1.8, 5.7] |
| France | 0.1681 | TimeXer | 0.1723 | 2.5 | [0.5, 4.5] |
| Denmark | 0.2404 | TimeXer | 0.2463 | 2.4 | [0.8, 4.0] |
| Portugal | 0.1953 | iTransformer | 0.2032 | 3.9 | [2.0, 5.7] |
| Austria | 0.2587 | TimeXer | 0.2653 | 2.5 | [1.0, 3.9] |
| Greece | 0.1787 | iTransformer | 0.1891 | 5.5 | [3.3, 7.7] |
| India (stress case, not in the average) | 0.0970 | iTransformer | 0.0969 | -0.1 | [-4.0, 3.5] |

Mean gain over the 11 main grids: 4.8% (range 2.4 to 8.4).

Mean per-target gain over the strongest baseline, 11 grids: load 6.2%, solar 8.6%, wind 2.3%

### Corrected protocol (every model predicts a correction on yesterday; GateXer vs corrected TimeXer)

| Grid | GateXer nMAE | Strongest baseline | Its nMAE | Gain % | 95% CI |
|---|---|---|---|---|---|
| ERCOT | 0.1474 | TimeXer | 0.1495 | 1.4 | [-0.8, 3.5] |
| CAISO | 0.1461 | TimeXer | 0.1461 | 0.1 | [-1.8, 1.9] |
| Germany | 0.1766 | TimeXer | 0.1776 | 0.6 | [-1.4, 2.6] |
| Spain | 0.1435 | TimeXer | 0.1436 | 0.1 | [-1.4, 1.6] |
| India (stress case, not in the average) | 0.0910 | TimeXer | 0.0914 | 0.5 | [-1.2, 2.1] |

### Which kind of sharing helps? (gain over the no-sharing control, 3-seed ensembles)

| Grid | Control | Full GateXer | Sources only | Weather only |
|---|---|---|---|---|
| ERCOT | GateXer without sharing | 5.7 | 2.9 | 5.1 |
| CAISO | GateXer without sharing | 3.8 | 1.8 | 2.4 |
| SWPP | ci (no sharing, no FAMO, no trend split) | 3.9 | 2.7 | 4.7 |
| MISO | ci (no sharing, no FAMO, no trend split) | 3.2 | 1.7 | 1.1 |
| Portugal | ci (no sharing, no FAMO, no trend split) | 1.8 | 2.0 | 1.9 |
| Germany | GateXer without sharing | 0.8 | 0.6 | 1.5 |
