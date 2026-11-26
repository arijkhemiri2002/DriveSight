# Stored GPU detections vs CPU FP32 reference

Frames re-run on CPU: 64 (6 sequences, every 5th sample)

- reference boxes (conf>=0.30) found in stored output (conf>=0.20): 100.00% of 1249
- stored boxes (conf>=0.30) found in reference (conf>=0.20): 100.00% of 1249

**Verdict: PASS - the stored detections reproduce the CPU FP32 reference**

```
                 ref_boxes  ref_found_in_stored  stored_boxes  stored_found_in_ref
seq                                                                               
test:25                169                  169           169                  169
test:6                 236                  236           236                  236
training:2             189                  189           189                  189
training:4             224                  224           224                  224
validation:1737        187                  187           187                  187
validation:1744        244                  244           244                  244
```
