### Tolerance Loss for Trimap Generation.

```
            F  B  U
IF GT:      0  0  1   => Must not be wrong
            1  0  0   => Must not be B, but can make U
            0  1  0   => Must not be F, but can make U

    F   B   U              √        ×
F   √   ×   √         F   F+U       B
B   ×   √   √    =>   B   B+U       F     
U   ×   ×   √         U    U       F+B
```

```
tolerance_loss = - 1/N \Sum_i^N [M_F\log(p_F+p_U)+M_B\log(p_F+p_U)+M_U\log(p_U)]

M_x means the mask of region x labeled in trimap.
p_x means the probability of region x (after softmax).
```

In the implementation, it can be converted into binary classification and then call cross entropy.

Loss = base_loss + **10** × tolerance_loss